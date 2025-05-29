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
from fish_speech.models.entro_vc.mask import make_pad_mask
from fish_speech.models.flow_decoder.length_regulator import InterpolateRegulator
from fish_speech.models.entro_vc.utils import lens_to_mask,mask_from_frac_lengths

class DeDup_DiTVC(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 mel_dim: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 frac_lengths_mask = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.mel_dim = mel_dim
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder
        self.frac_lengths_mask = frac_lengths_mask

        self.input_embedding = nn.Embedding(vocab_size, input_size, padding_idx=self.vocab_size-1)

        self.fill_emb = nn.Embedding(1, self.mel_dim)

        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), mel_dim) if self.encoder.output_size() != mel_dim else None


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask.unsqueeze(-1)

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()

        frac_lengths = torch.zeros((batch_size,), device=h.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(mel_feat_lens, frac_lengths)
        rand_span_mask &= mask
        conds = torch.where(rand_span_mask[..., None], torch.zeros_like(mel_feat.transpose(1,2)), mel_feat.transpose(1,2))
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h,
            cond=conds,
            loss_mask=rand_span_mask.unsqueeze(1),
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        token = self.input_embedding(torch.clamp(codes, min=0)) * mask.unsqueeze(-1)

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        token = self.input_embedding(torch.clamp(codes, min=0)) * mask.unsqueeze(-1)

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # # get conditions
        # feat = mel_feat.transpose(1,2)
        # conds = torch.zeros(feat.shape, device=mel_feat.device)
        # conds = conds.transpose(1, 2)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        for i, j in enumerate(mel_feat_lens):
            index = random.randint(int(0.3 * j), int(0.32 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_from_token(self,prompt_mel_feat,prompt_mel_feat_lens,codes,code_lengths,prompt_token,prompt_token_len,gen_mel_len,n_timesteps=10):

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[-1], codes.shape[-1]
        token, token_len = torch.concat([prompt_token, codes], dim=-1), prompt_token_len + code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(prompt_mel_feat)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        mel_len1, mel_len2 = prompt_mel_feat.shape[-1], gen_mel_len

        ### 填充fill embedding
        batch_size, seq_len = prompt_mel_feat.shape[0],(mel_len1+mel_len2)
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, prompt_mel_feat.shape[1]], device=token.device)
        conds[:, :mel_len1] = prompt_mel_feat.transpose(1,2)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=n_timesteps
        )
        return feat[:,:,mel_len1:]

class Entropy_DiTVC(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 mel_dim: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 frac_lengths_mask = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.mel_dim = mel_dim
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder
        self.frac_lengths_mask = frac_lengths_mask

        self.fill_emb = nn.Embedding(1, self.mel_dim)

        self.encoder_proj =  torch.nn.Linear(self.encoder.dim, mel_dim) if self.encoder.dim != mel_dim else None

    
    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h,
            cond=conds,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat
    
    @torch.inference_mode()
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat


    def forward1(self,mel_feat,mel_feat_lens,codes,code_lengths,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()

        frac_lengths = torch.zeros((batch_size,), device=h.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(mel_feat_lens, frac_lengths)
        rand_span_mask &= mask
        conds = torch.where(rand_span_mask[..., None], torch.zeros_like(mel_feat.transpose(1,2)), mel_feat.transpose(1,2))
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h,
            cond=conds,
            loss_mask=rand_span_mask.unsqueeze(1),
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference1(self,mel_feat,mel_feat_lens,codes,code_lengths,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        if self.encoder_proj is not None:
            h = self.encoder_proj(h)

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(codes))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=mel_feat.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_rec(self,codes,code_lengths):
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
            n_timesteps=10
        )
        return feat

    @torch.inference_mode()
    def inference_from_token(self,prompt_mel_feat,prompt_mel_feat_lens,codes,code_lengths,prompt_token,prompt_token_len,n_timesteps=10):

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
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=n_timesteps
        )
        return feat[:,:,mel_len1:]

class Entropy_DiTVC_Spk(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 decoder_dim: int = 512,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.fill_emb = nn.Embedding(1, self.decoder_dim)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, decoder_dim)

        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), decoder_dim) if self.encoder.output_size() != decoder_dim else None


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(h))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # get conditions, random mask
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[-1],1).transpose(1, 2).contiguous()
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h,
            cond=conds,
            spks=embedding
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,patch_ids,num_patches):
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross

        ### 填充fill embedding
        batch_size, seq_len = mel_feat.shape[0],mel_feat.shape[-1]
        h_len = h.shape[1]
        assert h_len<seq_len,'error length value'
        fill_embed = self.fill_emb(torch.zeros((batch_size,seq_len - h_len)).to(h))  
        h = torch.cat((h,fill_embed),dim=1).transpose(1,2).contiguous()   ## (B,C,T)

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(mel_feat_lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[-1],1).transpose(1, 2).contiguous()
        feat = self.decoder(
            mu=h,
            mask=mask.unsqueeze(1),
            cond=conds,
            spks=embedding,
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