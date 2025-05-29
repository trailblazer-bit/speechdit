# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import time
from collections import defaultdict
from enum import Enum

import torch
from pydantic import BaseModel
from torch.nn import functional as F


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 1
    data_loader_patching: bool = False
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False

    # def build(self) -> "Patcher":
    #     return Patcher(self)


def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy


def calculate_entropies(
    tokens: torch.tensor, entropy_model, patching_batch_size, device: str | None = None
):
    """
    tokens: 2D tensor of shape [batch_size, seq_len]
    Return 2D tensor of shape [batch_size, seq_len] with entropies for each token.

    Splits the tokens into chunks of size max_length and calculates entropies for each chunk.
    Entropy model can be executed on cpu or gpu, specify either 'cuda' or 'cpu' in the device argument.
    """
    with torch.no_grad():
        entropies = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            assert torch.all(split >= 0) and torch.all(split < 260)
            pred, _ = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)

        entropies = torch.cat(entropies, dim=0)
        entropies = entropies.reshape(tokens.shape)
    return entropies


def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids


def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    # assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths

def find_space_patch_start_ids(tokens):
    return patch_start_ids


def to_device(entropy_model, device=None):
    return entropy_model, device



def apply_bpe_patcher(tokens, bpe_patcher, patching_batch_size, device=None):
    return bpe_patching_mask




def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]

    # first_ids = (
    #     torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
    #     .unsqueeze(0)
    #     .repeat(bs, 1)
    # )
    # preds_truncation_len = first_ids.shape[
    #     1
    # ]  # remove the first preds because they will be start of patches.
    # entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        # patch_start_mask[1:] |= tokens[:-1] < OFFSET
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    # patch_start_ids = torch.cat(
    #     (first_ids, patch_start_ids + preds_truncation_len), dim=1
    # )
    return patch_start_ids


def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


def find_bpe_delim_patch_start_ids(tokens, delim):
    return patch_start_ids



def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst


class Patcher:
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.data_loader_patching = patcher_args.data_loader_patching
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time
        if self.log_time:
            self.log = defaultdict(float)

    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float = None,
        entropy_model=None,
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None

        # ENTROPY
        if self.log_time:
            s = time.time()
        if entropies is not None:
            scores = torch.tensor(entropies, dtype=torch.float32)
        elif preds is not None:
            scores = entropy(preds)
        else:
            start_entropies = time.time()
            scores = calculate_entropies(
                tokens,
                entropy_model,
                self.patching_batch_size,
                self.device,
            )
        if self.log_time:
            self.log["calculate_entropies"] += time.time() - s
            s = time.time()
        patch_start_ids = find_entropy_patch_start_ids(
            scores,
            self.patch_size,
            include_next_token=include_next_token,
            threshold=threshold if threshold is not None else self.threshold,
            threshold_add=self.threshold_add,
            monotonicity=self.monotonicity,
        )
        if self.log_time:
            self.log["find_entropy_patch_start_ids"] += time.time() - s
            s = time.time()
        patch_lengths = patch_lengths_from_start_ids(
            patch_start_ids, seq_len_next_tok
        )
        if self.log_time:
            self.log["patch_lengths_from_start_ids"] += time.time() - s
            s = time.time()


        # Apply any processing to patch lengths
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        # assert not check_non_zero_after_zero(patch_lengths)
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, : patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        assert (
            torch.sum(patch_lengths)
            == tokens.numel() + include_next_token * tokens.shape[0]
        ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        return patch_lengths, scores