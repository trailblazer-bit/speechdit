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

from fish_speech.models.dur_predictor.models.commons.llama import LLaMa
from fish_speech.models.dur_predictor.models.commons.vc_modules import ConvGlobalStacks


class SentenceDurPredictor(torch.nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp = hp
        self.dur_pred = LLaMa(self.hp)
        self.ph_proj = nn.Sequential(
            nn.Embedding(hp.n_phone+1, hp.phone_embed_dim, padding_idx=hp.n_phone),
            nn.Linear(hp.phone_embed_dim, hp.encoder_dim)
        )
        self.prompt_encoder = ConvGlobalStacks(
                idim=hp.in_channels, n_chans=hp.spk_e_dim, odim=hp.spk_embed_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)

    def sequence_mask(self, seq_lens, max_len=None, device='cpu', dtype=torch.float16):
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(seq_lens.device) # [1, t]
        mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
        mask = mask.to(dtype)
        return mask


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths) -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)        
        ph_emb = self.ph_proj(hubert_codes)  # [B, T, 1024]
        ph_lens = hubert_code_lengths  #[B]
        """ speaker embedding (global speaker reference) """
        T_mel = mels.shape[-1]

        start_idx = 0
        min_end_idx = int(T_mel * 0.4)
        max_end_idx = int(T_mel * 0.6)
        end_idx = random.randint(min_end_idx, max_end_idx)
        spk_mel = mels[:, :, start_idx:end_idx]

        spk_emb = self.prompt_encoder(spk_mel.transpose(1,2))          #[B,C=512]    
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ph_emb.shape[1], -1) #[B,T=T_text,C=512] (Repet To Text Time Dim)
        cond = ph_emb + spk_emb
        padding_mask = self.sequence_mask(ph_lens) > 0
        decoder_out = self.dur_pred(cond, padding_mask)
        pred_mel_len = self.postnet(decoder_out).squeeze(-1).sum(-1)

        dur_loss = F.mse_loss((pred_mel_len + 1).log(), (mel_lengths + 1).log())
        return {'dur_loss': dur_loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths):
        ## mels (B,C,T)        
        ph_emb = self.ph_proj(hubert_codes)  # [B, T, 1024]
        ph_lens = hubert_code_lengths  #[B]
        """ speaker embedding (global speaker reference) """
        T_mel = mels.shape[-1]

        start_idx = 0
        min_end_idx = int(T_mel * 0.4)
        max_end_idx = int(T_mel * 0.5)
        end_idx = random.randint(min_end_idx, max_end_idx)
        spk_mel = mels[:, :, start_idx:end_idx]

        spk_emb = self.prompt_encoder(spk_mel.transpose(1,2))          #[B,C=512]    
        spk_emb = spk_emb.unsqueeze(1).expand(-1, ph_emb.shape[1], -1) #[B,T=T_text,C=512] (Repet To Text Time Dim)
        cond = ph_emb + spk_emb
        padding_mask = self.sequence_mask(ph_lens) > 0
        decoder_out = self.dur_pred(cond, padding_mask)
        pred_mel_len = self.postnet(decoder_out).squeeze(-1).sum(-1)

        dur_loss = F.mse_loss((pred_mel_len + 1).log(), (mel_lengths + 1).log())
        return {'dur_loss': dur_loss}



