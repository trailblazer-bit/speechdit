# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn as nn
from torch.nn import functional as F
from xformers.ops import AttentionBias
from pydantic import BaseModel
import sys
sys.path.append('')

from fish_speech.models.entropy_llm.base_transformer import (
    InitStdFactor,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
    repeat_kv
)
from fish_speech.models.entropy_asr.asr_decoder import WhisperAttention,WhisperSdpaAttention

from fish_speech.models.entropy_asr.utils import create_causal_mask, downsample

logger = logging.getLogger()

class LocalEncoderArgs(BaseModel):
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_length: int = 0
    dropout: float = 0.0
    vocab_size: int = 0
    patch_size: int = None
    efficient_attn: str = None
    sliding_window: int = None
    use_rope: bool = True
    cross_attn_encoder: bool = True
    cross_attn_k: int = 8
    downsampling_by_pooling: str = "max"
    cross_attn_all_layers_encoder: bool = True
    cross_attn_init_by_pooling: bool = True
    cross_attn_nheads: int = 4
    fixed_length: int = 0

    # Local encoder specific dimensions
    n_heads_local_encoder: int = 8
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None


class CrossAttention(nn.Module):
    """
    CrossAttention block to attend to the encoder states from the decoder.
    Rope is not supported.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.cross_attn_norm_q = RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask = None,
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        assert mask is not None
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))

        # Standard attention computation
        attn_weights = torch.matmul(xq, xk.transpose(-1, -2))  # B H S D x B H D S -> B H S S
        if mask is not None:
            attn_weights += mask

        attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize over the sequence length dimension

        # Apply attention weights to values
        output = torch.matmul(attn_weights, xv)  # B H S S x B H S D -> B H S D

        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        output = self.wo(output.reshape(output_shape))

        return x + output

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std * factor

        nn.init.trunc_normal_(
            self.wq.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wk.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wv.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        output_std = std / (2**0.5)
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=output_std,
            a=-3 * output_std,
            b=3 * output_std,
        )
        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()


class LocalModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size

        self.efficient_attn = args.efficient_attn
        self.sliding_window = args.sliding_window
        self.use_rope = args.use_rope
        self.init_std_factor = args.init_std_factor
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim, padding_idx=self.vocab_size-1)
        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(args.max_length, args.dim)
        else:
            self.rope = RotaryEmbedding(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=getattr(args, "max_encoder_seq_length", args.max_length),
            )
            self.pos_embeddings = None

        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb") and args.dim_token_emb != self.dim
            else None
        )

        self.patch_embedding_projection = self._create_patch_projection(args)

    def _should_create_patch_projection(self, args):
        dimension_mismatch = (
            getattr(args, "dim_patch_emb") and args.dim_patch_emb != self.dim
        )

        # Check cross attention conditions
        cross_attn_conditions = (
            hasattr(args, "cross_attn_encoder")
            and args.cross_attn_encoder
            and getattr(args, "cross_attn_init_by_pooling")
        ) or (
            hasattr(args, "cross_attn_decoder")
            and args.cross_attn_decoder
            and getattr(args, "cross_attn_init_by_pooling")
        )

        cross_attn_conditions = None

        return dimension_mismatch or cross_attn_conditions

    def _create_patch_projection(self, args):
        if not self._should_create_patch_projection(args):
            return None

        output_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        return nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def init_weights(self, init_std=None):
        self.rope.reset_parameters()

        init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(
                self.pos_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(init_std, factor)

        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.patch_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.patch_embedding_projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if hasattr(self, "output"):
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                factor = {
                    InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                    InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                    InitStdFactor.DIM_RATIO: self.dim / 4096,
                    InitStdFactor.DISABLED: 1.0,
                }[self.init_std_factor]

                layer.init_weights(init_std, factor)



class LocalEncoder1(LocalModelBase):
    def __init__(self, args):
        super().__init__(args)
        # self.output_proj = (
        #     args.patching_mode in ["entropy", "probmax"]
        # ) and args.entropy_model_checkpoint_dir is None

        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads
        self.args = args

        if self.cross_attn_encoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            self.cross_attn_norms = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for layer_idx in range(layers_to_add):
                self.cross_attn_layers.append(
                    WhisperSdpaAttention(
                        embed_dim=self.dim,
                        num_heads=self.cross_attn_nheads,
                        dropout=args.dropout,
                        is_decoder=False,
                        layer_idx=layer_idx,
                        config=None
                    )
                )
                self.cross_attn_norms.append(nn.LayerNorm(self.dim))

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)
    
    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["AttentionBias", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen = tokens.shape
        if mask is None:
            mask = create_causal_mask(seqlen, self.efficient_attn, self.sliding_window)

        h = self.apply_embedding(tokens, embeds)
        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.efficient_attn)
            ### padding 部分mask,merge的时候比较关键
            h = h * cross_mask.unsqueeze(-1)
            # check if cross attention should be applied to either all layer or only the last layer
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0

        assert cross_mask.shape[-1] == h.shape[1]
        encoder_cross_mask = self.invert_attention_mask(cross_mask)

        # Cross-Attention Block
        if h is not None:
            residual = patch_embeds
            patch_embeds = self.cross_attn_norms[layer_idx](patch_embeds)
            patch_embeds_cross, _, _ = self.cross_attn_layers[layer_idx](
                hidden_states=patch_embeds,
                key_value_states=h,
                attention_mask=encoder_cross_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=None,
            )
            patch_embeds_cross = nn.functional.dropout(patch_embeds_cross, p=self.args.dropout, training=self.training)
            patch_embeds = residual + patch_embeds_cross

        return patch_embeds


class LocalEncoder_Fixed_Length(LocalModelBase):
    def __init__(self, args):
        super().__init__(args)
        # self.output_proj = (
        #     args.patching_mode in ["entropy", "probmax"]
        # ) and args.entropy_model_checkpoint_dir is None

        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads
        self.args = args
        self.fixed_length = args.fixed_length

        self.pooling_layer = nn.MaxPool1d(kernel_size=self.fixed_length)

        if self.cross_attn_encoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            self.cross_attn_norms = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for layer_idx in range(layers_to_add):
                self.cross_attn_layers.append(
                    WhisperSdpaAttention(
                        embed_dim=self.dim,
                        num_heads=self.cross_attn_nheads,
                        dropout=args.dropout,
                        is_decoder=False,
                        layer_idx=layer_idx,
                        config=None
                    )
                )
                self.cross_attn_norms.append(nn.LayerNorm(self.dim))

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)
    
    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["AttentionBias", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen = tokens.shape
        if mask is None:
            mask = create_causal_mask(seqlen, self.efficient_attn, self.sliding_window)

        h = self.apply_embedding(tokens, embeds)
        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.efficient_attn)
            ### padding 部分mask,merge的时候比较关键
            h = h * cross_mask.unsqueeze(-1)
            # check if cross attention should be applied to either all layer or only the last layer
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            if h.shape[1] % self.fixed_length != 0:
                hidden_states = torch.nn.functional.pad(h.transpose(1,2).contiguous(), (0, \
                    self.fixed_length - h.shape[1] % self.fixed_length))
            else:
                hidden_states = h.transpose(1,2).contiguous()
            patch_embeds = self.pooling_layer(hidden_states).permute(0, 2, 1)

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0

        assert cross_mask.shape[-1] == h.shape[1]
        encoder_cross_mask = self.invert_attention_mask(cross_mask)

        # Cross-Attention Block
        if h is not None:
            residual = patch_embeds
            patch_embeds = self.cross_attn_norms[layer_idx](patch_embeds)
            patch_embeds_cross, _, _ = self.cross_attn_layers[layer_idx](
                hidden_states=patch_embeds,
                key_value_states=h,
                attention_mask=encoder_cross_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=None,
            )
            patch_embeds_cross = nn.functional.dropout(patch_embeds_cross, p=self.args.dropout, training=self.training)
            patch_embeds = residual + patch_embeds_cross

        return patch_embeds

class LocalEncoder_Merge(nn.Module):
    def __init__(self, args, encoder_module: nn.Module):
        super().__init__()

        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads
        self.args = args
        self.patch_size = args.patch_size
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim, padding_idx=self.vocab_size-1)
        self.encoder_module = encoder_module

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        patch_lens: Optional[torch.Tensor] = None,
    ):
        """ """
        bs, seqlen = tokens.shape

        h = self.apply_embedding(tokens, embeds)

        ### padding 部分mask,merge的时候比较关键
        h = h * mask.unsqueeze(-1)

        patch_embeds = self.apply_merge_function(
            h, patch_embeds, bs, num_patches, patch_ids
        )

        h, h_lengths = self.encoder_module(patch_embeds, patch_lens)

        return h

    def apply_merge_function(
        self, h, patch_embeds, bs, num_patches, patch_ids
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
        return patch_embeds


class LocalEncoder(LocalModelBase):
    def __init__(self, args):
        super().__init__(args)
        # self.output_proj = (
        #     args.patching_mode in ["entropy", "probmax"]
        # ) and args.entropy_model_checkpoint_dir is None

        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        if self.cross_attn_encoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["AttentionBias", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen = tokens.shape
        if mask is None:
            mask = create_causal_mask(seqlen, self.efficient_attn, self.sliding_window)

        h = self.apply_embedding(tokens, embeds)
        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.efficient_attn)
            # check if cross attention should be applied to either all layer or only the last layer
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        patch_embeds += patch_embeds_cross
        return patch_embeds

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def test():
    args = LocalEncoderArgs(
        dim=512,
        n_layers=1,
        n_heads=8,
        max_length=12288,
        vocab_size=503,
        efficient_attn='sdpa',
        dim_token_emb=512,
        dim_patch_emb=512
    )
    model = LocalEncoder(args)

    total_params = count_parameters(model)

    print(f"Total number of parameters: {total_params}")

# test()