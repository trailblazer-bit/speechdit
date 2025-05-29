import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import List
import sys
sys.path.append('')

from fish_speech.models.flow_dit.modules.embedding import TokenEmbedding, SinePositionalEmbedding
from fish_speech.models.flow_dit.modules.convnet import ConvNetDouble, ConvNet
from fish_speech.models.flow_dit.modules.transformer import (TransformerEncoder,
                                 TransformerEncoderLayer,
                                 MultiHeadAttention)
from fish_speech.models.flow_dit.modules.utils import make_attn_mask


class MRTE(nn.Module):
    def __init__(
            self,
            mel_bins: int = 100,
            mel_activation: str = 'ReLU',
            mel_kernel_size: int = 3,
            mel_stride: int = 16,
            mel_n_layer: int = 5,
            mel_n_stack: int = 5,
            mel_n_block: int = 2,
            content_ff_dim: int = 1024,
            content_n_heads: int = 2,
            content_n_layers: int = 8,
            hidden_size: int = 512,
            token_vocab_size: int = 320,
            dropout: float = 0.1
    ):
        super(MRTE, self).__init__()

        self.n_heads = content_n_heads
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size

        self.token_embedding = TokenEmbedding(
            dim_model=hidden_size,
            vocab_size=token_vocab_size+1,
            dropout=dropout,
            padding_idx=token_vocab_size,
        )

        self.token_pos_embedding = SinePositionalEmbedding(
            dim_model=hidden_size,
            dropout=dropout,
        )

        self.mel_encoder_middle_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=mel_stride + 1,
            stride=mel_stride,
            padding=(mel_stride) // 2,
        )
        self.mel_encoder = ConvNetDouble(
            in_channels=mel_bins,
            out_channels=hidden_size,
            hidden_size=hidden_size,
            n_layers=mel_n_layer,
            n_stacks=mel_n_stack,
            n_blocks=mel_n_block,
            middle_layer=self.mel_encoder_middle_layer,
            kernel_size=mel_kernel_size,
            activation=mel_activation,
        )

        self.token_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=hidden_size,
                ff_dim=content_ff_dim,
                conv_ff=True,
                n_heads=content_n_heads,
                dropout=dropout,
            ),
            num_layers=content_n_layers,
        )

        self.mha = MultiHeadAttention(
            qkv_dim=hidden_size,
            n_heads=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()      

    def tc_latent(
            self,
            token: torch.Tensor,  # (B, T)
            token_lens: torch.Tensor,  # (B,)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        token_emb = self.token_embedding(token)
        token_pos = self.token_pos_embedding(token_emb)

        mel = rearrange(mel, 'B T D -> B D T')
        mel_context = self.mel_encoder(mel)
        mel_context = rearrange(mel_context, 'B D T -> B T D')
        token_x = self.token_encoder(token_pos)

        tc_latent = self.mha(token_x, kv=mel_context)
        tc_latent = self.norm(tc_latent)
        tc_latent = self.activation(tc_latent)

        return tc_latent

    def forward(
            self,
            token: torch.Tensor,  # (B, T)
            token_lens: torch.Tensor,  # (B,)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        tc_latent = self.tc_latent(token, token_lens, mel) # (B,T,C)
        return tc_latent


class MRTE_Text(nn.Module):
    def __init__(
            self,
            mel_activation: str = 'ReLU',
            mel_kernel_size: int = 3,
            mel_stride: int = 16,
            mel_n_layer: int = 5,
            mel_n_stack: int = 5,
            mel_n_block: int = 2,
            content_ff_dim: int = 1024,
            content_n_heads: int = 2,
            content_n_layers: int = 8,
            hidden_size: int = 512,
            dropout: float = 0.1,
            text_encoder: torch.nn.Module = None
    ):
        super(MRTE_Text, self).__init__()

        self.n_heads = content_n_heads
        self.hidden_size = hidden_size
        self.text_encoder = text_encoder

        self.token_pos_embedding = SinePositionalEmbedding(
            dim_model=hidden_size,
            dropout=dropout,
        )

        self.token_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=hidden_size,
                ff_dim=content_ff_dim,
                conv_ff=True,
                n_heads=content_n_heads,
                dropout=dropout,
            ),
            num_layers=content_n_layers,
        )

        self.mha = MultiHeadAttention(
            qkv_dim=hidden_size,
            n_heads=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()      

    def tc_latent(
            self,
            token_emb: torch.Tensor,  # (B, T, C)
            token_mask: torch.Tensor,  # (B,T)
            text_embed: torch.Tensor,  # (B, T, C)
            text_mask: torch.Tensor,  # (B,T)
            text_lengths: torch.Tensor,  # (B,)
    ):
        token_pos = self.token_pos_embedding(token_emb)

        text_context,_ = self.text_encoder(text_embed,text_lengths)
        token_x = self.token_encoder(token_pos) * token_mask

        tc_latent = self.mha(token_x*token_mask, kv=text_context*text_mask)+token_x
        tc_latent = self.norm(tc_latent)
        tc_latent = self.activation(tc_latent)

        return tc_latent

    def forward(
            self,
            token_emb: torch.Tensor,  # (B, T, C)
            token_mask: torch.Tensor,  # (B,T)
            text_embed: torch.Tensor,  # (B, T, C)
            text_mask: torch.Tensor,  # (B,T)
            text_lengths: torch.Tensor,  # (B,)
    ):
        tc_latent = self.tc_latent(token_emb, token_mask, text_embed, text_mask, text_lengths) # (B,T,C)
        return tc_latent


class MRTE_Mel(nn.Module):
    def __init__(
            self,
            mel_bins: int = 100,
            mel_activation: str = 'ReLU',
            mel_kernel_size: int = 3,
            mel_stride: int = 16,
            mel_n_layer: int = 5,
            mel_n_stack: int = 5,
            mel_n_block: int = 2,
            content_ff_dim: int = 1024,
            content_n_heads: int = 2,
            content_n_layers: int = 8,
            hidden_size: int = 512,
            token_vocab_size: int = 320,
            dropout: float = 0.1
    ):
        super(MRTE_Mel, self).__init__()

        self.n_heads = content_n_heads
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size

        self.token_pos_embedding = SinePositionalEmbedding(
            dim_model=hidden_size,
            dropout=dropout,
        )

        self.mel_encoder_middle_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=mel_stride + 1,
            stride=mel_stride,
            padding=(mel_stride) // 2,
        )
        self.mel_encoder = ConvNetDouble(
            in_channels=mel_bins,
            out_channels=hidden_size,
            hidden_size=hidden_size,
            n_layers=mel_n_layer,
            n_stacks=mel_n_stack,
            n_blocks=mel_n_block,
            middle_layer=self.mel_encoder_middle_layer,
            kernel_size=mel_kernel_size,
            activation=mel_activation,
        )

        self.token_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=hidden_size,
                ff_dim=content_ff_dim,
                conv_ff=True,
                n_heads=content_n_heads,
                dropout=dropout,
            ),
            num_layers=content_n_layers,
        )

        self.mha = MultiHeadAttention(
            qkv_dim=hidden_size,
            n_heads=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()      

    def tc_latent(
            self,
            token_embed: torch.Tensor,  # (B, T, C)
            token_lens: torch.Tensor,  # (B,)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        token_pos = self.token_pos_embedding(token_embed)

        mel = rearrange(mel, 'B T D -> B D T')
        mel_context = self.mel_encoder(mel)
        mel_context = rearrange(mel_context, 'B D T -> B T D')
        token_x = self.token_encoder(token_pos)

        tc_latent = self.mha(token_x, kv=mel_context) + token_x
        tc_latent = self.norm(tc_latent)
        tc_latent = self.activation(tc_latent)

        return tc_latent

    def forward(
            self,
            token_embed: torch.Tensor,  # (B, T, C)
            token_lens: torch.Tensor,  # (B,)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        tc_latent = self.tc_latent(token_embed, token_lens, mel) # (B,T,C)
        return tc_latent



def test():
    mrte = MRTE(
        mel_bins=100,
        mel_activation='ReLU',
        mel_kernel_size= 3,
        mel_stride=16,
        mel_n_layer=5,
        mel_n_stack= 5,
        mel_n_block= 2,
        content_ff_dim=1024,
        content_n_heads=2,
        content_n_layers=8,
        hidden_size=512,
        token_vocab_size=500,
        dropout=0.1
    )
    device = "cuda:0"
    mrte = mrte.to(device)


    t = torch.randint(0, 500, (2, 10)).to(dtype=torch.int64).to(device)
    tl = torch.tensor([6, 10]).to(dtype=torch.int64).to(device)
    m = torch.randn(2, 347, 100).to(device)

    out = mrte(t, tl, m)
    print(out.shape)