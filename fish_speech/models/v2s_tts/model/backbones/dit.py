"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from fish_speech.models.dit_vc.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

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


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        if spk_dim is None:
            self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        else:
            self.proj = nn.Linear(mel_dim * 3 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], spk_embeds=None,drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        if spk_embeds is None:
            x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        else:
            x = self.proj(torch.cat((x, cond, text_embed, spk_embeds), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class InputEmbedding_WoStyle(nn.Module):
    def __init__(self, mel_dim, style_dim, out_dim, spk_dim=None):
        super().__init__()
        if spk_dim is None:
            self.proj = nn.Linear(mel_dim * 2, out_dim)
        else:
            self.proj = nn.Linear(mel_dim * 3, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], token_embed: float["b n d"], spk_embeds=None,drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        if spk_embeds is None:
            x = self.proj(torch.cat((x, token_embed), dim=-1))
        else:
            x = self.proj(torch.cat((x, token_embed, spk_embeds), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class InputEmbedding_Style(nn.Module):
    def __init__(self, mel_dim, style_dim, out_dim, spk_dim=None):
        super().__init__()
        if spk_dim is None:
            self.proj = nn.Linear(mel_dim * 2, out_dim)
        else:
            self.proj = nn.Linear(mel_dim * 3, out_dim)
        self.style_proj = nn.Linear(style_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], token_embed: float["b n d"], spk_embeds=None,drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        
        style_cond = self.style_proj(cond)

        if spk_embeds is None:
            x = self.proj(torch.cat((x, token_embed), dim=-1))
        else:
            x = self.proj(torch.cat((x, token_embed, spk_embeds), dim=-1))
        x = torch.cat((style_cond,x),dim=1)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        spk_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        if spk_dim is None:
            self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        else:
            self.input_embed = InputEmbedding(mel_dim, text_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x,   #  float["b d n"],  # nosied input audio  # noqa: F722
        cond, #  float["b d n"],  # masked cond audio  # noqa: F722
        token_embed, #  int["b d n"],  # text  # noqa: F722
        time, # float["b"] | float[""],  # time step  # noqa: F821 F722
        mask, # bool["b n"],  # noqa: F722
        spk_embeds = None,
        drop_audio_cond = False,  # cfg for cond audio
        drop_text = False,  # cfg for text
    ):
        x = x.transpose(1,2).contiguous()
        cond = cond.transpose(1,2).contiguous()
        token_embed = token_embed.transpose(1,2).contiguous()
        if spk_embeds is not None:
            spk_embeds = spk_embeds.transpose(1,2).contiguous()
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if spk_embeds is None: 
            x = self.input_embed(x, cond, token_embed, drop_audio_cond=drop_audio_cond)
        else:
            x = self.input_embed(x, cond, token_embed,spk_embeds=spk_embeds,drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1,2)  # (B,C,T)

        return output


class DiT_Style(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        style_dim=None,
        spk_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if spk_dim is None:
            self.input_embed = InputEmbedding_Style(mel_dim, style_dim, dim)
        else:
            self.input_embed = InputEmbedding_Style(mel_dim, style_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x,   #  float["b d n"],  # nosied input audio  # noqa: F722
        cond, #  float["b d n"],  # masked cond audio  # noqa: F722
        token_embed, #  int["b d n"],  # text  # noqa: F722
        time, # float["b"] | float[""],  # time step  # noqa: F821 F722
        mask, # bool["b n"],  # noqa: F722
        spk_embeds = None,
        drop_audio_cond = False,  # cfg for cond audio
        drop_text = False,  # cfg for text
    ):
        x = x.transpose(1,2).contiguous()
        cond = cond.transpose(1,2).contiguous()
        token_embed = token_embed.transpose(1,2).contiguous()
        if spk_embeds is not None:
            spk_embeds = spk_embeds.transpose(1,2).contiguous()
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if spk_embeds is None: 
            x = self.input_embed(x, cond, token_embed, drop_audio_cond=drop_audio_cond)
        else:
            x = self.input_embed(x, cond, token_embed,spk_embeds=spk_embeds,drop_audio_cond=drop_audio_cond)
        
        style_mask = torch.ones((cond.shape[0],cond.shape[1])).to(mask)
        mask = torch.cat((style_mask,mask),dim=1)
        style_len = cond.shape[1]
        seq_len = x.shape[1]

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1,2)  # (B,C,T)

        return output[:,:,style_len:]


class DiT_WoStyle(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        style_dim=None,
        spk_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if spk_dim is None:
            self.input_embed = InputEmbedding_WoStyle(mel_dim, style_dim, dim)
        else:
            self.input_embed = InputEmbedding_WoStyle(mel_dim, style_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x,   #  float["b d n"],  # nosied input audio  # noqa: F722
        cond, #  float["b d n"],  # masked cond audio  # noqa: F722
        token_embed, #  int["b d n"],  # text  # noqa: F722
        time, # float["b"] | float[""],  # time step  # noqa: F821 F722
        mask, # bool["b n"],  # noqa: F722
        spk_embeds = None,
        drop_audio_cond = False,  # cfg for cond audio
        drop_text = False,  # cfg for text
    ):
        x = x.transpose(1,2).contiguous()
        token_embed = token_embed.transpose(1,2).contiguous()
        if spk_embeds is not None:
            spk_embeds = spk_embeds.transpose(1,2).contiguous()
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if spk_embeds is None: 
            x = self.input_embed(x, cond, token_embed, drop_audio_cond=drop_audio_cond)
        else:
            x = self.input_embed(x, cond, token_embed,spk_embeds=spk_embeds,drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1,2)  # (B,C,T)

        return output