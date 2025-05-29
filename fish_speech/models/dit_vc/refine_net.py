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


def mask_segments(x, ids_str, segment_size,mask):
    ret = torch.zeros_like(x[:, :, :segment_size])
    new_mask = mask.clone().to(mask.device)
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
        new_mask[i,idx_str:idx_end]=0
    return ret,new_mask

def rand_mask_segments(x, x_lengths=None,mask=None, min_ratio=0.2, max_ratio=0.5):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    
    x_lengths = x_lengths.min()
    # Ensure min_ratio and max_ratio are between 0 and 1
    min_ratio = max(0, min(min_ratio, 1))
    max_ratio = max(0, min(max_ratio, 1))

    # Calculate min and max segment sizes
    min_segment_size = int(min_ratio * x_lengths)
    max_segment_size = int(max_ratio * x_lengths)

    # Randomly choose a segment size within the range
    segment_size = torch.randint(min_segment_size, max_segment_size + 1, (1,)).item()

    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ids_str = torch.max(torch.zeros(ids_str.size()).to(ids_str.device), ids_str).to(
        dtype=torch.long
    )
    seg_x, new_mask = mask_segments(x, ids_str, segment_size,mask)
    mask_x = x * new_mask.unsqueeze(1)
    return seg_x, mask_x


  

class RefineNet_DiTVC(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
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
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths):
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

class RefineNet_DiTVC_Spk_ShortCut(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size, input_size,padding_idx=self.vocab_size-1)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, output_size)

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=conds,
            spks=embedding
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            spks=embedding,
            n_timesteps=4
        )
        return feat
    
    @torch.inference_mode()
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds,n_timesteps=4):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

        # get conditions
        conds = torch.zeros(mel_feat.shape, device=token.device)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            spks=embedding,
            n_timesteps=n_timesteps
        )
        return feat

    @torch.inference_mode()
    def inference_from_token(self,prompt_mel_feat,prompt_mel_feat_lens,codes,code_lengths,prompt_token,prompt_token_len,spk_embeds,n_timesteps=10):

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[-1], codes.shape[-1]
        token, token_len = torch.concat([prompt_token, codes], dim=-1), prompt_token_len + code_lengths
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(prompt_mel_feat)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

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

class RefineNet_DiTVC_WoSpk_ShortCut(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size, input_size,padding_idx=self.vocab_size-1)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=4
        )
        return feat
    
    @torch.inference_mode()
    def inference_zero(self,mel_feat,mel_feat_lens,codes,code_lengths,n_timesteps=4):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

        # get conditions
        # conds = torch.zeros(mel_feat.shape, device=token.device)
        # get conditions
        feat = mel_feat.transpose(1,2)
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(mel_feat_lens):
            index = random.randint(int(0.3 * j), int(0.32 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(mel_feat_lens)).to(h).bool()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            cond=conds,
            n_timesteps=n_timesteps
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

class RefineNet_DiTVC_Spk(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 output_type: str = "mel",
                 vocab_size: int = 500,
                 spk_dim: int = 192,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, output_size)

        self.lr = InterpolateRegulator(output_size,sampling_ratios=[1,1,1,1])


    def forward(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds) -> Dict[str, Optional[torch.Tensor]]:
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        loss, _ = self.decoder.compute_loss(
            mel_feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond=conds,
            spks=embedding
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,mel_feat,mel_feat_lens,codes,code_lengths,spk_embeds):
        # concat text and prompt_text
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(mel_feat)
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # text encode
        h, h_lengths = self.encoder(token, code_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, mel_feat_lens)

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
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1).transpose(1, 2).contiguous()
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
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