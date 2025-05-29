"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding


from fish_speech.models.entro_vc.modules import ConvNeXtV2Block,precompute_freqs_cis,get_pos_embed_indices
# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds, text_dim,padding_idx=text_num_embeds-2) 

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 1000  
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text):  # noqa: F722
        batch, seq_len = text.shape[0], text.shape[1]

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


class TextEmbedding_Pad(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_num_embeds = text_num_embeds
        self.text_embed = nn.Embedding(text_num_embeds, text_dim,padding_idx=text_num_embeds-2) 

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 1000  
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len):  # noqa: F722
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=self.text_num_embeds-3)  ### filter_token

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text



# text_encoder = TextEmbedding(255,1024,conv_layers=4)
# a = torch.randint(100,(16,80))
# out = text_encoder(a)
# print(out.shape)