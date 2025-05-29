import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from modules.tts.commons.align_ops import expand_states
from modules.tts.megatts3.flow_matching.llama import LLaMa
from modules.voice_conversion.vc_modules import ConvGlobalStacks
from modules.tts.megatts3.flow_matching.time_embedding import DDPMTimestepEmbedder, CFMTimeEmbedding
from modules.tts.megatts3.frontend_lm import add_prefix
import torchdiffeq
from diffusers import DDPMScheduler, DDIMScheduler

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    # frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 128
    bpe_embed_dim: int = 384

    n_phone: int = 300 + 2
    n_tone: int = 30 + 2
    n_bpe: int = 32008
    bpe_pad: int = 32005

    local_cond_dim: int = 512
    time_embed_dim: int = 256

    local_cond_project_type: str = "linear"  # conv
    local_cond_conv_kernel: int = 9
    local_cond_conv_padding: int = 4

    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 8192
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    dropout: float = 0.0
    ffn_dim_multiplier: Optional[float] = None
    use_causal_attn: bool = False

    causal: bool = False
    use_qk_norm: str = ""  # head, channel
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda: [-1, -1])
    window_type: str = "elemwise"  # elemwise, blockwise
    llama_provider: str = "ctiga"

    # speaker encoder
    spk_e_dim: int = 1024
    spk_embed_dim: int = 512

    # postnet
    postnet_type: str = "linear"  # conv
    postnet_kernel: int = 3

    target: str = "bn"
    prompt_feature: str = "bn"
    in_channels: int = 16
    out_channels: int = 16
    use_textprefix: bool = True

    bias: bool = False
    target_type: str = "vector_field"

    # for uniform t
    min_t: float = 0.0
    max_t: float = 1.0
    lognormal_mean: float = 0.0
    lognormal_std: float = 1.0

    p_max_t: float = 0.0

    use_seg_embed: bool = True
    use_bn_eos_bos: bool = True

    max_phone_len: int = 2000
    max_bn_len: int = 4000

    flashattn_version: str = "2.3"

    use_expand_ph: bool = False
    zero_xt_prompt: bool = False

    use_causal_attn: bool = False
    
    use_qk_norm: bool = False
    use_cache: bool = False
    max_cache_batch_size: int = 10


class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = hp.min_t if hasattr(hp, "min_t") else 0
        self.max_t = hp.max_t if hasattr(hp, "max_t") else 1

        self.target_type = hp.target_type if hasattr(hp, "target_type") else "vector_field"

        self.act_fn = nn.GELU()
        self.bias = hp.bias

        # text.
        self.ph_proj = nn.Sequential(
            nn.Embedding(hp.n_phone, hp.phone_embed_dim, padding_idx=0),
            nn.Linear(hp.phone_embed_dim, hp.encoder_dim)
        )
        self.tone_proj = nn.Sequential(
            nn.Embedding(hp.n_tone, hp.tone_embed_dim, padding_idx=0),
            nn.Linear(hp.tone_embed_dim, hp.encoder_dim)
        )
        if hp.use_bpe:
            self.bpe_proj = nn.Sequential(
                nn.Embedding(hp.n_bpe, hp.bpe_embed_dim, padding_idx=hp.bpe_pad),
                nn.Linear(hp.bpe_embed_dim, hp.encoder_dim)
            )

        # time-embedding
        self.time_embedding = CFMTimeEmbedding(hp.time_embed_dim, bias=self.bias)
        if self.target_type == 'epsilon':
            self.time_embedding = DDPMTimestepEmbedder(hp.time_embed_dim)

        # global speaker-embedding
        self.prompt_encoder = ConvGlobalStacks(
                idim=hp.in_channels, n_chans=hp.spk_e_dim, odim=hp.spk_embed_dim)
        
        local_cond_in_channels = hp.out_channels + hp.spk_embed_dim

        self.local_cond_project = nn.Linear(
            local_cond_in_channels, hp.local_cond_dim, bias=self.bias)

        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]
            
        self.encoder = LLaMa(hp)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim, bias=self.bias)
        self.prenet = nn.Linear(
            hp.time_embed_dim + hp.local_cond_dim, hp.encoder_dim, bias=self.bias
        )
        if hp.use_expand_ph:
            self.expand_ph_prenet = nn.Linear(
                hp.encoder_dim, hp.encoder_dim, bias=self.bias
            )

        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
  
        self.use_seg_embed = hp.use_seg_embed
        if hp.use_seg_embed:
            self.seg_embed = nn.Embedding(3, hp.encoder_dim, padding_idx=0)
            nn.init.trunc_normal_(self.seg_embed.weight, std=0.02, a=-0.04, b=0.04)

        if hp.use_bn_eos_bos:
            self.bn_eos_bos = nn.Parameter(torch.randn(2, hp.encoder_dim))  
        
        if self.target_type == 'epsilon':
            self.noise_scheduler = DDPMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                num_train_timesteps=1000, rescale_betas_zero_snr=False, clip_sample=False)
        elif self.target_type == 'vector_field':
            ''' sigma==0 means ODE'''
            from modules.tts.megatts3.flow_matching.vp_cfm import VariancePreservingConditionalFlowMatcher
            self.flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.0)
        else:
            raise NotImplementedError
        
        # self.prefix_cfg_embed = nn.Parameter(torch.randn(dim // 2))

    def forward(self, inputs, sigmas=None, x_noisy=None):
        """ text projection """
        ph_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        ph_lens, feat_lens = inputs["text_lens"], inputs["lat_lens"]

        ctx_feature = inputs['lat'] * inputs['ctx_mask']
        B, device = ctx_feature.size(0), ctx_feature.device

        """ speaker embedding (global speaker reference) """
        spk_emb = self.prompt_encoder(ctx_feature) # prompt encoder can automatically deal with paddings
        spk_emb = spk_emb * (1 - inputs["spk_cfg_mask"])
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ctx_feature.shape[1], -1)

        """ local conditioning (prompt_latent + spk_embed) """
        ctx_feature = ctx_feature * (1 - inputs["spk_cfg_mask"][:, :, None])
        local_cond = torch.cat([ctx_feature, spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        """ diffusion target latent """
        x = inputs['lat']

        if self.target_type == "epsilon":
            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=x.device,
                dtype=torch.int64
            )
            # time embedding
            time_emb = self.time_embedding(timesteps)
            time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
            # define noisy_input and target
            x_noisy = self.noise_scheduler.add_noise(x, noise, timesteps)
            target = noise
        elif self.target_type == "vector_field":
            # Here, x is x1 in CFM
            x0 = torch.randn_like(x)
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
            # time embedding
            time_emb = self.time_embedding(t)
            time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
            # define noisy_input and target
            x_noisy = xt
            target = ut
        else:
            raise NotImplementedError

        # concat condition.
        if self.hp.zero_xt_prompt:
            x_noisy = x_noisy * (1 - inputs['ctx_mask'])
        x_noisy = self.x_prenet(x_noisy) + self.prenet(
            torch.cat([time_emb, local_cond], dim=-1)
        )
        if self.hp.use_expand_ph:
            rand_num = random.random()
            if rand_num <= 0.33:
                x_noisy += expand_states(ph_embed, inputs['mel2ph'])
            elif rand_num <= 0.67:
                x_noisy += expand_states(ph_embed, inputs['sparsified_dur'])
            else:
                pass
        
        # Add BPE tokens
        if self.hp.use_bpe:
            bpe_embed, bpe_lens = self.bpe_proj(inputs["bpe"]), inputs["bpe_lengths"]
            text_embed = add_prefix(bpe_embed, bpe_lens, ph_embed, ph_lens)
            text_lens = ph_lens + bpe_lens
        else:
            text_embed = ph_embed
            text_lens = ph_lens

        if self.hp.use_bn_eos_bos:
            inputs["text_mel_mask"] = F.pad(inputs["text_mel_mask"], (2, 0), "constant", 1)
            feat_lens = feat_lens + 2
            bn_bos = self.bn_eos_bos[0][None, None, :].expand(x_noisy.shape[0], -1, -1)
            x_noisy = torch.cat([bn_bos, x_noisy, torch.zeros_like(bn_bos)], dim=1)
            indics_x = torch.arange(x_noisy.shape[1], device=device)[None, :]
            mask_eos = (indics_x == feat_lens[:, None]-1)
            x_noisy[mask_eos] = self.bn_eos_bos[1][None, None, :]

        # concat prefix-text.
        if self.hp.use_textprefix:
            T = inputs["text_mel_mask"].shape[1]
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full(
                [B, T, C], 0, device=device, dtype=x_noisy.dtype
            )
            if self.use_seg_embed:
                seg_input = torch.zeros([B, T], device=x_noisy.device).long()

            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = (
                torch.arange(text_embed.shape[1], device=device)[None, :]
                < text_lens[:, None]
            )
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)
            if self.use_seg_embed:
                seg_input[mask_x] = 1

            mask_x = (
                (text_lens[:, None] <= indics_x)
                & (indics_x < (text_lens + feat_lens)[:, None])
                & (indics_x - text_lens[:, None] < T_feat)
            )
            mask_noisy = (
                torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            )
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)
            if self.use_seg_embed:
                seg_input[mask_x] = 2

            encoder_input = x_noisy_wtext
            seq_mask = inputs["text_mel_mask"]
            if self.use_seg_embed:
                seg_output = self.seg_embed(seg_input)
                encoder_input = encoder_input + seg_output
        else:
            encoder_input = x_noisy
            seq_mask = inputs[f"{self.hp.prompt_feature}_mask"]

        encoder_out = self.encoder(encoder_input, seq_mask)

        if self.hp.use_bn_eos_bos:
            feat_lens = feat_lens - 2

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(B, x.shape[1], encoder_out.shape[-1], device=device)

            T0 = x.shape[1]
            T1 = encoder_out.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & (
                (text_lens[:, None] + indics0) < T1
            )
            if self.hp.use_bn_eos_bos:
                mask1 = (text_lens[:, None] < indics1) & (
                    indics1 < (text_lens[:, None] + feat_lens[:, None]+1)
                )
            else:
                mask1 = (text_lens[:, None] <= indics1) & (
                    indics1 < (text_lens[:, None] + feat_lens[:, None])
                )

            pred_v_wotext[mask0] = encoder_out[mask1].to(dtype=pred_v_wotext.dtype)

            pred_v = pred_v_wotext

        pred = self.postnet(pred_v)

        ret_dict = {
            "pred_v": pred.transpose(1, 2),
            "target_v": target.transpose(1, 2),
        }
        return ret_dict

    def _forward(self, x, local_cond, text_embed, bpe_embed, timesteps, ctx_mask, dur=None, cfg_w=1.0, use_seq_cfg=False, seq_cfg_w=[1.0,1.0]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(local_cond.shape[0], local_cond.shape[1], -1)
        if self.hp.zero_xt_prompt:
            x = x * (1 - ctx_mask)
        x = self.x_prenet(x) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        if dur is not None:
            x += expand_states(text_embed, dur)
        
        # Add BPE tokens
        if self.hp.use_bpe:
            text_embed = torch.cat([bpe_embed, text_embed], dim=1)

        if self.hp.use_bn_eos_bos:
            bn_bos = self.bn_eos_bos[0][None, None, :].expand(x.shape[0], -1, -1)
            bn_eos = self.bn_eos_bos[1][None, None, :].expand(x.shape[0], -1, -1)
            x = torch.cat([bn_bos, x, bn_eos], dim=1)

        if self.hp.use_textprefix:
            x = torch.cat([text_embed, x], dim=1)
            if self.use_seg_embed:
                seg_input = torch.zeros([1, x.shape[1]], device=x.device).long()
                seg_input[:, :text_embed.shape[1]] = 1
                seg_input[:, text_embed.shape[1]:] = 2
                seg_output = self.seg_embed(seg_input)
                x = x + seg_output

        pred_v = self.encoder(x, attn_mask=torch.ones((x.size(0), x.size(1)), device=x.device))

        if self.hp.use_textprefix:
            pred_v = pred_v[:, text_embed.shape[1] :, :]

        if self.hp.use_bn_eos_bos:
            pred_v = pred_v[:, 1:-1, :]

        pred = self.postnet(pred_v)
        
        """ Perform CFG """
        if cfg_w != 1:
            if use_seq_cfg:
                cond_spk_txt, cond_txt, uncond = pred.chunk(3)
                pred = uncond + seq_cfg_w[0] * (cond_txt - uncond) \
                    + seq_cfg_w[1] * (cond_spk_txt - cond_txt)
                # pred = cond_spk_txt + seq_cfg_w[1]*(cond_spk_txt - cond_txt) + seq_cfg_w[0]*(cond_txt - uncond)
            else:
                cond_spk_txt, uncond = pred.chunk(2)
                pred = uncond + cfg_w * (cond_spk_txt - uncond)
        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, cfg_w=1.0, use_seq_cfg=False, seq_cfg_w=[1.0, 1.0], **kwargs):
        # txt embedding
        text_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        if self.hp.use_bpe:
            bpe_embed = self.bpe_proj(inputs['bpe'])
        else:
            bpe_embed = None

        # speaker embedding
        ctx_feature = inputs['lat_ctx']
        spk_emb = self.prompt_encoder(ctx_feature)
        if cfg_w != 1:
             spk_emb[1:, :] = 0 # enable global spk cfg
             ctx_feature[1:, :, :] = 0 # prefix spk cfg
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ctx_feature.shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([ctx_feature, spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        if self.target_type == 'epsilon':
            # Build scheduler
            scheduler = DDIMScheduler.from_config(
                self.noise_scheduler.config,
                rescale_betas_zero_snr=False, clip_sample=False, set_alpha_to_one=False, thresholding=False)
            scheduler.set_timesteps(timesteps)

            _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
            x = x * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        x, local_cond, text_embed, bpe_embed, timesteps=torch.LongTensor([t]).to(x.device).repeat(2)
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, text_embed, bpe_embed, timesteps=torch.LongTensor([t]).to(x.device).repeat(2)
                    )
                x = scheduler.step(v_pred, t, x).prev_sample

        elif self.target_type == 'vector_field':
            bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            
            if cfg_w != 1:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(torch.cat([x] * bsz), local_cond, text_embed, bpe_embed, timesteps=t.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], cfg_w=cfg_w, use_seq_cfg=use_seq_cfg, seq_cfg_w=seq_cfg_w),
                    torch.randn([1, frm_len, self.hp.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                )
            else:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(x, local_cond, text_embed, bpe_embed, timesteps=t.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur']),
                    torch.randn([1, frm_len, self.hp.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
            x = traj[-1]
        
        else:
            raise NotImplementedError

        return x
