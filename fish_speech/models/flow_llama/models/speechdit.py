import logging
import math
import random
from dataclasses import dataclass, field, fields
import dataclasses
from math import pi
from typing import Sequence, Tuple, Union, Optional

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from models.commons.align_ops import expand_states
from models.commons.llama import LLaMa
from models.commons.vc_modules import ConvGlobalStacks
import torchdiffeq
from diffusers import DDPMScheduler, DDIMScheduler

logger = logging.getLogger(__name__)


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class RMSNorm(nn.Module):
    def __init__(self, dim, feat_dim=-1, eps=1e-5):
        super().__init__()
        self.rms = dim**-0.5
        self.feat_dim = feat_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, unscaled=False):
        norm = torch.norm(x, dim=self.feat_dim, keepdim=True) * self.rms
        if unscaled:
            return x / norm.clamp(min=self.eps)
        g = self.scale
        if self.feat_dim != -1:
            while g.ndim <= self.feat_dim:
                g = g[None]
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
        return x / norm.clamp(min=self.eps) * g


"""For VDiffusion"""


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin

class LogitNormalDistribution(Distribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean, self.std = mean, std

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        x = torch.from_numpy(np.random.lognormal(self.mean, self.std, num_samples)).to(device)
        return x / (1 + x)

class BernoulliDistribution(Distribution):
    def __init__(self, v1, v2):
        super().__init__()
        self.map = torch.tensor([v1, v2]).unsqueeze(0)

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        index = (torch.rand(num_samples, device=device) > 0.5).long()
        return self.map.repeat(num_samples, 1).to(device)[
            torch.arange(num_samples), index
        ]



def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def Ts(t):
    """Builds a type template for a given type that accepts a list of instances"""
    return lambda *types: lambda: t(*[tp() for tp in types])


class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Repeat(m, times: int):
    ms = (m,) * times
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)


class TimeEmbedding(nn.Module):
    def __init__(self, modulation_features, num_layers: int = 2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features, bias=bias),
                nn.GELU(),
            ),
            times=num_layers,
        )

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


def get_sigma(t: Tensor):
    angle = t * math.pi / 2
    sigma = torch.tan(angle)
    return sigma


@dataclass
class ModelArgs:
    # frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 128
    #n_phone: int = 300 + 2
    n_phone: int = 74 + 2

    n_tone: int = 30 + 2

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
    in_channels: int = 24
    out_channels: int = 24
    #in_channels: int = 8
    #out_channels: int = 8

    use_textprefix: bool = True

    bias: bool = False
    target_type: str = "vector_field" 
    #target_type: str = "epsilon"


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
    use_bpe: bool=False


class Diffusion(nn.Module):
    def __init__(self, hp, hp_from_file):
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
        #self.tone_proj = nn.Sequential(
        #    nn.Embedding(hp.n_tone, hp.tone_embed_dim, padding_idx=0),
        #    nn.Linear(hp.tone_embed_dim, hp.encoder_dim)
        #)

        #if hp.use_bpe:
        #    self.bpe_proj = nn.Sequential(
        #        nn.Embedding(hp.n_bpe, hp.bpe_embed_dim, padding_idx=hp.bpe_pad),
        #        nn.Linear(hp.bpe_embed_dim, hp.encoder_dim)
        #    )


        # time-embedding
        #self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)
        from models.commons.time_embedding import DDPMTimestepEmbedder, CFMTimeEmbedding
        self.time_embedding = CFMTimeEmbedding(hp.time_embed_dim, bias=self.bias)
        if self.target_type == 'epsilon':
            self.time_embedding = DDPMTimestepEmbedder(hp.time_embed_dim)

        self.in_channels = in_channels = hp_from_file["vae_channel"] if hp_from_file.get("use_mel",False) == False else 160
        self.out_channels = out_channels = hp_from_file["vae_channel"] if hp_from_file.get("use_mel",False) == False else 160
        # global speaker-embedding
        self.prompt_encoder = ConvGlobalStacks(
                idim=in_channels, n_chans=hp.spk_e_dim, odim=hp.spk_embed_dim)
        
        local_cond_in_channels = out_channels + hp.spk_embed_dim

        self.local_cond_project = nn.Linear(
            local_cond_in_channels, hp.local_cond_dim, bias=self.bias)

        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]

        ## backbone
        #@dataclass
        #class LLaMaArgs:
        #    dim: int = 1024
        #    n_layers: int = 24
        #    n_heads: int = 16
        #    n_kv_heads: Optional[int] = None
        #    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        #    norm_eps: float = 1e-5
        #    max_seq_len: int = 16384
        #    dropout: float = 0.0
        #    ffn_dim_multiplier: Optional[float] = None
        #    use_causal_attn: bool = True
        #self.encoder = LLaMa(LLaMaArgs())
        self.encoder = LLaMa(hp)


        self.x_prenet = nn.Linear(in_channels, hp.encoder_dim, bias=self.bias)
        self.prenet = nn.Linear(
            hp.time_embed_dim + hp.local_cond_dim, hp.encoder_dim, bias=self.bias
        )
        if hp.use_expand_ph:
            self.expand_ph_prenet = nn.Linear(
                hp.encoder_dim, hp.encoder_dim, bias=self.bias
            )

        self.postnet = nn.Linear(hp.encoder_dim, out_channels, bias=False)
  
        #self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)

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
            #from modules.tts.ps2.flow_matching.vp_cfm import VariancePreservingConditionalFlowMatcher
            from torchcfm import VariancePreservingConditionalFlowMatcher
            self.flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.0)
        else:
            raise NotImplementedError

    def forward(self, inputs, sigmas=None, x_noisy=None):
        """ text projection """
        #ph_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        ph_embed = self.ph_proj(inputs["phone"])  # [B, T, 1024]
        ph_lens, feat_lens = inputs["text_lens"], inputs["lat_lens"]  #[B]

        ctx_feature = inputs['lat'] * inputs['ctx_mask']    #[B,T,C=8(latent dim)]
        B, device = ctx_feature.size(0), ctx_feature.device

        """ speaker embedding (global speaker reference) """
        spk_emb = self.prompt_encoder(ctx_feature)          #[B,C=512]     
        spk_emb = spk_emb * (1 - inputs["spk_cfg_mask"])
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ctx_feature.shape[1], -1) #[B,T,C=512] (Repet To Latent Time Dim)

        """ local conditioning (prompt_latent + spk_embed) """
        ctx_feature = ctx_feature * (1 - inputs["spk_cfg_mask"][:,:,None])
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
            #t = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            #time_emb = self.time_embedding(t)
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
            torch.cat([time_emb, local_cond], dim=-1)   #[B,T,C=1024]
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
            mask_eos = (indics_x == feat_lens[:, None]-1)   #把noise中的padding部分的随机化
            x_noisy[mask_eos] = self.bn_eos_bos[1][None, None, :]
            #breakpoint()

        # concat prefix-text.
        if self.hp.use_textprefix:
            T = inputs["text_mel_mask"].shape[1]
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full(
                [B, T, C], 0, device=device, dtype=x_noisy.dtype
            )
            if self.use_seg_embed:  #True
                seg_input = torch.zeros([B, T], device=x_noisy.device).long()

            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)      #mask-Text: Text的部分(T)都是1
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
            )       #Padding Mask
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

        #LLAMA Encode
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

    #def _forward(self, x, local_cond, text_embed, timesteps, ctx_mask=None, use_torchdiffeq=False, torchdiffeq_cfg=3.5, use_rescale=False):
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
        """
        if use_torchdiffeq:
            if use_rescale:
                weight = 4
                rescale = 1
                pos, neg = pred[0:1], pred[1:2]
                # Apply regular classifier-free guidance.
                cfg = neg + weight * (pos - neg)
                # Calculate standard deviations.
                std_pos = pos.std([1,2], keepdim=True)
                std_cfg = cfg.std([1,2], keepdim=True)
                # Apply guidance rescale with fused operations.
                factor = std_pos / std_cfg
                factor = rescale * factor + (1 - rescale)
                pred = cfg * factor
            else:
                pred = torchdiffeq_cfg * pred[0:1] + (1 - torchdiffeq_cfg) * pred[1:2]
        """
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
        #text_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        text_embed = self.ph_proj(inputs["phone"])  # [B, T, 1024]
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
            #breakpoint()
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ctx_feature.shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([ctx_feature, spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        if self.target_type == 'epsilon':
            # Build scheduler
            scheduler = DDIMScheduler.from_config(
                self.noise_scheduler.config,
                rescale_betas_zero_snr=False, clip_sample=False, set_alpha_to_one=False, thresholding=False)
            scheduler.set_timesteps(50)

            t = 50
            _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            x = torch.randn([1, frm_len, self.out_channels], device=device)
            x = x * scheduler.init_noise_sigma

            #sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
            #sigmas = repeat(sigmas, "i -> i b", b=1)

            for i in scheduler.timesteps:
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        x, local_cond, text_embed, bpe_embed, timesteps=torch.LongTensor([t]).to(x.device).repeat(2)
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, text_embed,bpe_embed, timesteps=torch.LongTensor([t]).to(x.device).repeat(2)
                    )
                x = scheduler.step(v_pred, t, x).prev_sample

        elif self.target_type == 'vector_field':
            #t = timesteps
            bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            
            if cfg_w != 1:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(torch.cat([x] * bsz), local_cond, text_embed, bpe_embed, timesteps=t.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=None, cfg_w=cfg_w, use_seq_cfg=use_seq_cfg, seq_cfg_w=seq_cfg_w),
                    torch.randn([1, frm_len, self.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                )
            else:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(x, local_cond, text_embed, bpe_embed, timesteps=t.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=None),
                    torch.randn([1, frm_len, self.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
            x = traj[-1]
        
        else:
            raise NotImplementedError

        return x