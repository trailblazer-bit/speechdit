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
import math
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import random
from fish_speech.models.flow_decoder.mask import make_pad_mask
from fish_speech.models.vits_decoder.modules import modules,commons
from fish_speech.models.flow_decoder.length_regulator import InterpolateRegulator
from fish_speech.models.flow_decoder.modules import rand_slice_segments

from einops import rearrange


def create_alignment(base_mat, duration_tokens):
    N, L = duration_tokens.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_tokens[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_tokens[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator from FastSpeech """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        duration_tokens: torch.Tensor,  # (B, T) int for duration
    ):

        bsz, input_len, _ = x.size()

        expand_max_len = torch.max(torch.sum(duration_tokens, -1), -1)[0].int()

        alignment = torch.zeros(bsz, expand_max_len, input_len).numpy()
        alignment = create_alignment(alignment, duration_tokens.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)
        output = alignment @ x
        return output



class Refine_FlowDit_Style(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.ref_enc = modules.MelStyleEncoder(
            output_size, style_vector_dim=spk_embed_dim
        )
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.style_lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = mel_mask.unsqueeze(1)  # [B,1,T]
        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask) # (B,C,1)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = style_h.transpose(1, 2).contiguous(),
            spks = embedding.squeeze(1),
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = mel_mask.unsqueeze(1)  # [B,1,T]
        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask) # (B,C,1)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding.squeeze(1),
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = mel_mask.unsqueeze(1)  # [B,1,T]
        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask) # (B,C,1)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        feat_len = (style_code_lengths / 12.5 * 24000 / 256).int()
        style_h, _ = self.style_lr(style_h, feat_len)
        gen_mask = torch.ones((style_h.shape[0],style_h.shape[1])).unsqueeze(1).to(style_h)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding.squeeze(1),
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat



class Refine_FlowDit_Style_Mean_SV(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.style_lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = style_h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        # feat_len = (hubert_code_lengths / 50 * 24000 / 256).int()
        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        feat_len = (style_code_lengths / 12.5 * 24000 / 256).int()
        style_h, _ = self.style_lr(style_h, feat_len)
        gen_mask = torch.ones((style_h.shape[0],style_h.shape[1])).unsqueeze(1).to(style_h)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat


class Refine_FlowDit_Style_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.style_lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = style_h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        style_h, _ = self.style_lr(style_h, mel_lengths)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        feat_len = (style_code_lengths / 12.5 * 24000 / 256).int()
        style_h, _ = self.style_lr(style_h, feat_len)
        gen_mask = torch.ones((style_h.shape[0],style_h.shape[1])).unsqueeze(1).to(style_h)

        cond = h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        feat = self.decoder(
            mu=style_h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat


class Refine_FlowDit_Style_Prepend_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        feat_len = (hubert_code_lengths / 50 * 24000 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat

class Refine_FlowDit_Style_Prepend_HifiMel_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        feat_len = (hubert_code_lengths / 50 * 22050 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)

        cond = style_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(style_code_lengths)).to(cond).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat

class Refine_FlowDit_Style_Add_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])
        self.style_lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,mel_lengths)

        # get conditions
        conds = torch.zeros(style_h.shape, device=style_h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = style_h[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = conds,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,mel_lengths)

        # get conditions
        conds = torch.zeros(style_h.shape, device=style_h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = style_h[i, :index]
        conds = conds.transpose(1, 2)  # (B,C,T)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_rec(self,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        feat_len = (hubert_code_lengths / 50 * 22050 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,feat_len)

        conds = style_h.transpose(1, 2)  # (B,C,T)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            style_codes,style_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = int(token_len1 / 50 * 22050 / 256), int(token_len2 / 50 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,torch.tensor([mel_len1]).to(hubert_code_lengths))

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = style_h
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_MelMaskCond_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = conds,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = prompt_mel_lengths.max(), int(token_len2 / 50 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mels.transpose(1,2)
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_MelMaskCond_Mean_SV_DBatch_Mhubert(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = conds,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = prompt_mel_lengths.max(), int(token_len2 / 25 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mels.transpose(1,2)
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_DeDup_MelMaskCond_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = conds,
            spks = embedding,
            cond = h.transpose(1, 2).contiguous(),
            cond_mask = h_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=conds,
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=h.transpose(1, 2).contiguous(),
            cond_mask = h_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = prompt_mel_lengths.max(), int(token_len2 / 50 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mels.transpose(1,2)
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        # feat = feat[:, :, mel_len1:]
        # assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_DeDup__Mhubert_MelMaskCond_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = conds,
            spks = embedding,
            cond = h.transpose(1, 2).contiguous(),
            cond_mask = h_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(mels).unsqueeze(1)

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=conds,
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=h.transpose(1, 2).contiguous(),
            cond_mask = h_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = prompt_mel_lengths.max(), int(token_len2 / 25 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mels.transpose(1,2)
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_MelMaskCond_Mean_SV_DBatch_Wi_F0(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.style_encoder = style_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.style_encoder_out =  torch.nn.Linear(self.style_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])
        self.style_lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,mel_lengths)
        h_mu = torch.cat((h,style_h),dim=-1) 

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)


        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h_mu.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = conds,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        style_ssl = self.style_embedding(style_codes) 
        style_h, _ = self.style_encoder(style_ssl, style_code_lengths)
        style_h = self.style_encoder_out(style_h)
        style_h, _ = self.style_lr(style_h,mel_lengths)
        h_mu = torch.cat((h,style_h),dim=-1) 

        # get conditions
        feat = mels.transpose(1,2)
        conds = torch.zeros(feat.shape, device=h.device)
        for i, j in enumerate(mel_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h_mu.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,prompt_hubert,prompt_hubert_lens,\
            spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_hubert.shape[1], hubert_codes.shape[1]
        token, token_len = torch.concat([prompt_hubert, hubert_codes], dim=1), prompt_hubert_lens + hubert_code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.semantic_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_out_proj(h) # (B,T,C)

        mel_len1, mel_len2 = prompt_mel_lengths.max(), int(token_len2 / 50 * 22050 / 256)
        h, _ = self.lr.inference(h[:, :token_len1], h[:, token_len1:], torch.tensor([mel_len1]).to(hubert_code_lengths), torch.tensor([mel_len2]).to(hubert_code_lengths))
        gen_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_mels.transpose(1,2)
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=conds,
            n_timesteps=10
        )
        # feat = feat[:, :, mel_len1:]
        # assert feat.shape[2] == mel_len2
        return feat

class Refine_FlowDit_HubertOnly_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 style_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 style_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = None,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=None,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_rec(self,hubert_codes,hubert_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        feat_len = (hubert_code_lengths / 50 * 22050 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=None,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,prompt_mels,prompt_mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds):
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        feat_len = (hubert_code_lengths / 50 * 22050 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=None,
            n_timesteps=10
        )
        return feat

class Refine_FlowDit_Text_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 text_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 text_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.text_embedding = nn.Embedding(text_vocab, embed_size,padding_idx=0)
        self.text_encoder_out =  torch.nn.Linear(self.text_encoder.output_size(), output_size)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        text_h = self.text_embedding(texts) 
        text_h, _ = self.text_encoder(text_h, text_lengths)
        text_h = self.text_encoder_out(text_h)

        cond = text_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(text_lengths)).to(text_h).unsqueeze(1)

        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            cond = cond,
            cond_mask = cond_mask,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)

        h, _ = self.lr(h, mel_lengths)

        text_h = self.text_embedding(texts) 
        text_h, _ = self.text_encoder(text_h, text_lengths)
        text_h = self.text_encoder_out(text_h)

        cond = text_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(text_lengths)).to(text_h).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h = self.encoder_out_proj(h) # (B,T,C)
        
        feat_len = (hubert_code_lengths / 50 * 24000 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        text_h = self.text_embedding(texts) 
        text_h, _ = self.text_encoder(text_h, text_lengths)
        text_h = self.text_encoder_out(text_h)

        cond = text_h.transpose(1, 2).contiguous() # (B,C,T)
        cond_mask = (~make_pad_mask(text_lengths)).to(text_h).unsqueeze(1)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            cond=cond,
            cond_mask=cond_mask,
            n_timesteps=10
        )
        return feat


class Refine_FlowDit_CrossText_Mean_SV_DBatch(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 spk_embed_dim: int = 256,
                 output_type: str = "mel",
                 semantic_vocab: int = 500,
                 text_vocab: int = 48,
                 only_mask_loss: bool = True,
                 hop_length: int = 160,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 mrte_encoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.mrte_encoder = mrte_encoder
        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.text_embedding = nn.Embedding(text_vocab, embed_size,padding_idx=0)
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds) \
        -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)

        text_h = self.text_embedding(texts) 
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)
        text_mask = (~make_pad_mask(text_lengths)).to(h).unsqueeze(-1)
        h = self.mrte_encoder(h,h_mask,text_h,text_mask,text_lengths)

        h = self.encoder_out_proj(h) # (B,T,C)
        h, _ = self.lr(h, mel_lengths)

        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            mu = h.transpose(1, 2).contiguous(),
            spks = embedding,
            loss_mask=None,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)

        text_h = self.text_embedding(texts) 
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)
        text_mask = (~make_pad_mask(text_lengths)).to(h).unsqueeze(-1)
        h = self.mrte_encoder(h,h_mask,text_h,text_mask,text_lengths)

        h = self.encoder_out_proj(h) # (B,T,C)
        h, _ = self.lr(h, mel_lengths)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks = embedding,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        spk_embeds = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(spk_embeds)

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)

        text_h = self.text_embedding(texts) 
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)
        text_mask = (~make_pad_mask(text_lengths)).to(h).unsqueeze(-1)
        h = self.mrte_encoder(h,h_mask,text_h,text_mask,text_lengths)

        h = self.encoder_out_proj(h) # (B,T,C)
        feat_len = (hubert_code_lengths / 50 * 24000 / 256).int()
        h, _ = self.lr(h, feat_len)
        gen_mask = torch.ones((h.shape[0],h.shape[1])).unsqueeze(1).to(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=gen_mask,
            spks = embedding,
            n_timesteps=10
        )
        return feat