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
from fish_speech.models.flow_decoder.mask import make_pad_mask
from fish_speech.models.vits_decoder.modules import modules,commons
from fish_speech.models.flow_decoder.length_regulator import InterpolateRegulator
from fish_speech.models.flow_decoder.modules import DurationPitchPredictor,rand_slice_segments


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


class RefineNet(torch.nn.Module):
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
                 duration_predictor: torch.nn.Module = None,
                 prompt_encoder: torch.nn.Module = None
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
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.encoder_in_proj = torch.nn.Linear(embed_size*2, input_size)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.duration_predictor = duration_predictor
        self.semantic_lr = LengthRegulator()
        self.lr = InterpolateRegulator(self.encoder.output_size(),sampling_ratios=[1,1,1,1])

        self.prompt_encoder = prompt_encoder
        self.prompt_proj_in = torch.nn.Linear(output_size, input_size)
        


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations) -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)
        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2) ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align, 训练时duration使用gt
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)


        # mask = (~make_pad_mask(mel_feat_lens)).to(embedding)
        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding.squeeze(dim=1),
            loss_mask=loss_mask.unsqueeze(1),
        )
        return {'loss': loss,'dur_loss':dur_loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2)  ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        # h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss

    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2)  ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss

class RefineNet_Sdim(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 160,
                 embed_size: int = 256,
                 hubert_dim: int = 256,
                 style_dim: int = 128,
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
                 duration_predictor: torch.nn.Module = None,
                 prompt_encoder: torch.nn.Module = None
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
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.semantic_embedding = nn.Embedding(semantic_vocab+1, hubert_dim,padding_idx=semantic_vocab)
        self.style_embedding = nn.Embedding(style_vocab+1, style_dim,padding_idx=style_vocab)
        self.encoder_in_proj = torch.nn.Linear(hubert_dim+style_dim, input_size)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.duration_predictor = duration_predictor
        self.semantic_lr = LengthRegulator()
        self.lr = InterpolateRegulator(self.encoder.output_size(),sampling_ratios=[1,1,1,1])

        self.prompt_encoder = prompt_encoder
        self.prompt_proj_in = torch.nn.Linear(output_size, input_size)
        


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations) -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)
        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2) ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align, 训练时duration使用gt
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        # mask = (~make_pad_mask(mel_feat_lens)).to(embedding)
        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding.squeeze(dim=1),
            loss_mask=loss_mask.unsqueeze(1)
        )
        return {'loss': loss,'dur_loss':dur_loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2)  ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        # h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss

    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2)  ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        # h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss

class RefineNewNet(torch.nn.Module):
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
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 duration_predictor: torch.nn.Module = None,
                 prompt_encoder: torch.nn.Module = None
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
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length

        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.style_embedding = nn.Embedding(style_vocab+1, embed_size,padding_idx=style_vocab)
        self.encoder_in_proj = torch.nn.Linear(embed_size*2, input_size)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.duration_predictor = duration_predictor
        self.semantic_lr = LengthRegulator()
        self.lr = InterpolateRegulator(self.encoder.output_size(),sampling_ratios=[1,1,1,1])

        self.prompt_encoder = prompt_encoder
        self.prompt_proj_in = torch.nn.Linear(output_size, input_size)
        


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations) -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)
        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2) ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align, 训练时duration使用gt
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        prior_loss = torch.sum(0.5 * ((mels - h.transpose(1, 2).contiguous()) ** 2 + math.log(2 * math.pi)) * loss_mask.unsqueeze(1))
        prior_loss = prior_loss / (torch.sum(loss_mask) * self.output_size)

        # mask = (~make_pad_mask(mel_feat_lens)).to(embedding)
        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding.squeeze(dim=1),
            loss_mask=loss_mask.unsqueeze(1)
        )
        return {'loss': loss,'dur_loss':dur_loss,'prior_loss':prior_loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,
                style_codes,style_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        semantic_embed = self.semantic_embedding(hubert_codes) 
        style_embed = self.style_embedding(style_codes) 
        ssl = torch.cat([semantic_embed,style_embed],dim=2)  ## (B,T,C)
        ssl = self.encoder_in_proj(ssl)
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        # h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        prior_loss = torch.sum(0.5 * ((mels - h.transpose(1, 2).contiguous()) ** 2 + math.log(2 * math.pi)) * loss_mask.unsqueeze(1))
        prior_loss = prior_loss / (torch.sum(loss_mask) * self.output_size)


        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss,prior_loss

class RefineNet_Wo_Style(torch.nn.Module):
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
                 duration_predictor: torch.nn.Module = None,
                 prompt_encoder: torch.nn.Module = None
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
        self.only_mask_loss = only_mask_loss
        self.hop_length = hop_length
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.semantic_embedding = nn.Embedding(semantic_vocab+1, embed_size,padding_idx=semantic_vocab)
        self.encoder_out_proj =  torch.nn.Linear(self.encoder.output_size(), output_size)
        self.duration_predictor = duration_predictor
        self.semantic_lr = LengthRegulator()
        self.lr = InterpolateRegulator(self.encoder.output_size(),sampling_ratios=[1,1,1,1])

        self.prompt_encoder = prompt_encoder
        self.prompt_proj_in = torch.nn.Linear(output_size, input_size)
        


    def forward(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,durations) -> Dict[str, Optional[torch.Tensor]]:
        ## mels (B,C,T)
        
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h).unsqueeze(-1)  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align, 训练时duration使用gt
        h = self.semantic_lr(h,durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        # mask = (~make_pad_mask(mel_feat_lens)).to(embedding)
        loss, _ = self.decoder.compute_loss(
            mels,
            mel_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding.squeeze(dim=1),
            loss_mask=loss_mask.unsqueeze(1),
        )
        return {'loss': loss,'dur_loss':dur_loss}

    @torch.inference_mode()
    def inference(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels,_,_, loss_mask = rand_slice_segments(mels,mel_lengths,mel_mask,min_ratio=self.min_rate,max_ratio=self.max_rate)
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss

    @torch.inference_mode()
    def inference_one(self,mels,mel_lengths,hubert_codes,hubert_code_lengths,durations):
        mel_mask = (~make_pad_mask(mel_lengths)).to(mels)
        prompt_mels = mels
        prompt_mask = ((prompt_mels.abs().sum(dim=1)) > 0).float().unsqueeze(1)  # [B,1,T]

        ge = self.ref_enc(prompt_mels * prompt_mask, prompt_mask)
        embedding = self.spk_embed_affine_layer(ge.transpose(1,2).contiguous())

        ssl = self.semantic_embedding(hubert_codes) 
        h, h_lengths = self.encoder(ssl, hubert_code_lengths)
        h_mask = (~make_pad_mask(hubert_code_lengths)).to(h)[:,:,None]  # (B,T,1)

        ## duration
        prompt_lens = prompt_mask.squeeze(1).sum(-1)
        prompt_e = self.prompt_proj_in(prompt_mels.transpose(1,2).contiguous())  ## (B,T,C)
        prompt_embed,_ = self.prompt_encoder(prompt_e,prompt_lens) ## (B,T,C)
        pred_durations = self.duration_predictor(h,prompt_embed,x_mask=h_mask)
        dur_loss = F.mse_loss((pred_durations + 1).log(), (durations + 1).log(), reduction='none')
        dur_loss = (dur_loss * h_mask.squeeze(-1)).sum() / h_mask.squeeze(-1).sum()

        ## align
        h = self.semantic_lr(h,pred_durations) # (B,T,C)
        h, h_lengths = self.lr(h, mel_lengths)
        h = self.encoder_out_proj(h)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding.squeeze(dim=1),
            n_timesteps=10
        )
        return feat,dur_loss