# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from omegaconf import DictConfig
from fish_speech.models.flow_decoder.mask import make_pad_mask
from fish_speech.models.vits_decoder.modules import commons
from fish_speech.models.flow_decoder.length_regulator import InterpolateRegulator
from fish_speech.models.ssl_models.WavLM import WavLM, WavLMConfig
from fish_speech.models.ssl_models.utils.espnet_utils import make_pad_mask as get_pad_mask
from fish_speech.models.v2s_tts.style_bank import StyleBankExtractor

  
class V2S_TTS_Pretrain_Model(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 sll_checkpoint: str=None,
                 output_layer: int = 6,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_qformer: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size+1, input_size, padding_idx=vocab_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, output_size)

        ## load ssl model
        checkpoint = torch.load(sll_checkpoint)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.speech_encoder = WavLM(cfg)
        self.speech_encoder.load_state_dict(checkpoint['model'])
        self.speech_encoder.eval()
        for p in self.speech_encoder.parameters():
            p.requires_grad = False
        self.output_layer = output_layer

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])
        self.style_qformer = style_qformer
    
    def extract_batch_tensor(self, wav_tensor, frame_lens):
        # suppose wav is already a tensor padded with 0
        # should be careful with LayerNorm since it may cause difference between batch vs single modes.
        pad_mask = get_pad_mask(frame_lens).to(wav_tensor.device)
        with torch.no_grad():
            rep = self.speech_encoder.extract_features(wav_tensor, output_layer=self.output_layer, padding_mask=pad_mask)[0]
            return rep.clone().detach()  # [B, T, D]


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,wavs,wav_lengths) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        ## extract basic style codes
        style_q = self.extract_batch_tensor(wavs,wav_lengths)
        lens_q = wav_lengths//320
        mask_q = make_pad_mask(lens_q).float().to(mel_feat)
        if style_q.shape[1]>mask_q.shape[-1]:
            style_q = style_q[:,:mask_q.shape[-1],:]
        else:
            mask_q = mask_q[:,:style_q.shape[1]]
        style_embeds,_ = self.style_qformer(memory=style_q.transpose(0,1),padding_mask=mask_q)
        style_embeds = style_embeds.transpose(0,1).contiguous() # (B,T,C)

        # xvec projection
        # embedding = F.normalize(spk_embeds, dim=1)
        # embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)
        conds = style_embeds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,wavs,wav_lengths):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        style_q = self.extract_batch_tensor(wavs,wav_lengths)
        lens_q = wav_lengths//320
        mask_q = make_pad_mask(lens_q).float().to(mel_feat)
        if style_q.shape[1]>mask_q.shape[-1]:
            style_q = style_q[:,:mask_q.shape[-1],:]
        else:
            mask_q = mask_q[:,:style_q.shape[1]]
        style_embeds,_ = self.style_qformer(memory=style_q.transpose(0,1),padding_mask=mask_q)
        style_embeds = style_embeds.transpose(0,1).contiguous() # (B,T,C)


        # xvec projection
        # embedding = F.normalize(spk_embeds, dim=1)
        # embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)
        conds = style_embeds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_rec(self,codes,code_lengths,spk_embeds):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        feat_len = (code_lengths / 50 * 22050 / 256).int()
        h, h_lengths = self.lr(h, feat_len)

        # get conditions
        conds = torch.zeros(h.shape, device=token.device)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h).bool()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            spks=embedding,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds):
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        feat_len = (code_lengths / 50 * 22050 / 256).int()
        h, h_lengths = self.lr(h, feat_len)

        # get conditions
        # feat = mel_feat.transpose(1,2)
        # conds = torch.zeros(feat.shape, device=token.device)
        # for i, j in enumerate(mel_feat_lens):
        #     if random.random() < 0.5:
        #         continue
        #     index = random.randint(0, int(0.3 * j))
        #     conds[i, :index] = feat[i, :index]
        # conds = conds.transpose(1, 2)
        conds = torch.zeros(h.shape, device=token.device)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(embedding)
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_from_token(self,prompt_mel_feat,prompt_mel_feat_lens,codes,code_lengths,prompt_token,prompt_token_len,spk_embeds,n_timesteps=10):
        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[-1], codes.shape[-1]
        token, token_len = torch.concat([prompt_token, codes], dim=-1), prompt_token_len + code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(prompt_mel_feat)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_mel_feat.shape[-1], int(token_len2 / 50 * 22050 / 256)
        h, h_lengths = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]), torch.tensor([mel_len2]))

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mel_feat.transpose(1,2)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).bool()
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            spks=embedding,
            n_timesteps=n_timesteps
        )
        return feat[:,:,mel_len1:]


class V2S_TTS_Pretrain_Model_Wo_SBank(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 sll_checkpoint: str=None,
                 output_layer: int = 6,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_qformer: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size+1, input_size, padding_idx=vocab_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        # self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, output_size)

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,wavs,wav_lengths) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        # embedding = F.normalize(spk_embeds, dim=1)
        # embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=None
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,wavs,wav_lengths):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        # embedding = F.normalize(spk_embeds, dim=1)
        # embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=None,
            n_timesteps=10
        )
        return feat