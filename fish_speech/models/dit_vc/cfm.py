"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from fish_speech.models.dit_vc.model.utils import (
    default,
    exists,
    lens_to_mask,
    mask_from_frac_lengths,
)
from fish_speech.models.flow_decoder.mask import make_pad_mask
from fish_speech.models.vits_decoder.modules import commons
from fish_speech.models.flow_decoder.length_regulator import InterpolateRegulator


class CFM(nn.Module):
    def __init__(
        self,
        transformer_model: nn.Module,
        encoder: torch.nn.Module = None,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        vocab_size=501,
        input_size=512,
        num_channels=None,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),

    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # num_mels
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer_model
        dim = transformer_model.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        self.vocab_size = vocab_size
        self.encoder = encoder
        self.input_embedding = nn.Embedding(vocab_size, input_size,padding_idx=self.vocab_size-1)
        self.lr = InterpolateRegulator(num_channels,sampling_ratios=[1,1,1,1])
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), num_channels)



    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"],  # noqa: F722
        text_lens: int["b"],
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        # cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
    
        # text
        mask = (~make_pad_mask(text_lens)).float().unsqueeze(-1).to(cond)
        token = self.input_embedding(torch.clamp(text, min=0)) * mask
        h, h_lengths = self.encoder(token, text_lens)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, lens)
        text = h


        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        # get conditions
        step_cond = torch.zeros(cond.shape, device=cond.device)
        for i, j in enumerate(lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            step_cond[i, :index] = cond[i, :index]

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # max_duration = duration.amax()

        # cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        # if no_ref_audio:
        #     cond = torch.zeros_like(cond)

        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, cond, torch.zeros_like(cond)
        # )  # allow direct control (cut cond audio) with lens passed in


        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, token_embed=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, token_embed=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, out, cond)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"],  # noqa: F722
        text_lens: int["b"],
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        mask = (~make_pad_mask(text_lens)).float().unsqueeze(-1).to(inp)
        text = self.input_embedding(torch.clamp(text, min=0)) * mask
        h, h_lengths = self.encoder(text, text_lens)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, lens)
        text = h
        # if isinstance(text, list):
        #     if exists(self.vocab_char_map):
        #         text = list_str_to_idx(text, self.vocab_char_map).to(device)
        #     else:
        #         text = list_str_to_tensor(text).to(device)
        #     assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random.random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random.random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, token_embed=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred



class CFM_Spk(nn.Module):
    def __init__(
        self,
        transformer_model: nn.Module,
        encoder: torch.nn.Module = None,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        vocab_size=501,
        input_size=512,
        num_channels=None,
        spk_dim=None,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),

    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # num_mels
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer_model
        dim = transformer_model.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        self.vocab_size = vocab_size
        self.encoder = encoder
        self.input_embedding = nn.Embedding(vocab_size, input_size,padding_idx=self.vocab_size-1)
        self.lr = InterpolateRegulator(num_channels,sampling_ratios=[1,1,1,1])
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), num_channels)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_dim, num_channels)


    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"],  # noqa: F722
        text_lens: int["b"],
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        spk_embeds=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        # cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
    
        # text
        mask = (~make_pad_mask(text_lens)).float().unsqueeze(-1).to(cond)
        token = self.input_embedding(torch.clamp(text, min=0)) * mask
        h, h_lengths = self.encoder(token, text_lens)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, lens)
        text = h

        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        # get conditions
        step_cond = torch.zeros(cond.shape, device=cond.device)
        for i, j in enumerate(lens):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            step_cond[i, :index] = cond[i, :index]

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # max_duration = duration.amax()

        # cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        # if no_ref_audio:
        #     cond = torch.zeros_like(cond)

        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, cond, torch.zeros_like(cond)
        # )  # allow direct control (cut cond audio) with lens passed in


        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, token_embed=text, time=t, mask=mask,spk_embeds=embedding,drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, token_embed=text, time=t, mask=mask,spk_embeds=embedding, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, cond, out)
        out = torch.where(cond_mask, out, cond)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"],  # noqa: F722
        text_lens: int["b"],
        *,
        spk_embeds=None,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        mask = (~make_pad_mask(text_lens)).float().unsqueeze(-1).to(inp)
        text = self.input_embedding(torch.clamp(text, min=0)) * mask
        h, h_lengths = self.encoder(text, text_lens)
        h = self.encoder_proj(h)
        h, h_lengths = self.lr(h, lens)
        text = h
        # xvec projection
        embedding = F.normalize(spk_embeds, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1).repeat(1,h.shape[1],1)


        # if isinstance(text, list):
        #     if exists(self.vocab_char_map):
        #         text = list_str_to_idx(text, self.vocab_char_map).to(device)
        #     else:
        #         text = list_str_to_tensor(text).to(device)
        #     assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random.random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random.random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, token_embed=text, time=time,spk_embeds=embedding,drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred