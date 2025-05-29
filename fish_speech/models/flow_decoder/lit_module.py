from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn


from fish_speech.models.vqgan.utils import (
    avg_with_mask,
    plot_mel,
    sequence_mask
)
from fish_speech.third_party.vocoder_infer import BigVGAN

class FlowDecoder(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 320,
        sample_rate: int = 24000,
        mel_min: float = -1,
        mel_max: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        
        self.mel_min = mel_min
        self.mel_max = mel_max

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        self.vocoder = BigVGAN(next(self.generator.parameters()).device)


    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            }
        )

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)

        ret = self.generator(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,
                                style_codes,style_code_lengths,durations)

        gen_loss = ret['loss']
        dur_loss = ret['dur_loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/dur_loss",
            dur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss * self.lambda_gen + dur_loss * self.lambda_dur

        # Backward
        optim_g.zero_grad()

        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Manual LR Scheduler
        scheduler_g = self.lr_schedulers()
        scheduler_g.step()

    def validation_step(self, batch: Any, batch_idx: int):
        if batch_idx >= 2:
            return
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels, dur_loss = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,
                                style_codes,style_code_lengths,durations)
        ### denormalize
        refine_mels = refine_mels*(self.mel_max - self.mel_min)+ self.mel_min

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]

        refine_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, refine_mels, reduction="none"), mel_masks
        )

        self.log(
            "val/refine_mel_loss",
            refine_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/dur_loss",
            dur_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.spec2wav(gt_mels)
        gen_audios = self.vocoder.spec2wav(refine_mels)

        for idx, (
            mel,
            refine_mel,
            mel_len,
            gt_audio,
            gen_audio
        ) in enumerate(
            zip(
                gt_mels,
                refine_mels,
                mel_lengths,
                gt_audios.cpu().float(),
                gen_audios.cpu().float()
            )
        ):
            if idx > 4:
                break
            audio_len = mel_len * self.hop_length

            image_mels = plot_mel(
                [
                    refine_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Refine (Flow)",
                    "Ground-Truth",
                ],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        "wavs": [
                            wandb.Audio(
                                gt_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="gt",
                            ),
                            wandb.Audio(
                                gen_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="recon",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    gt_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gen",
                    gen_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)


class FlowNewDecoder(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 320,
        sample_rate: int = 24000,
        mel_min: float = -1,
        mel_max: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
        lambda_prior: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        
        self.mel_min = mel_min
        self.mel_max = mel_max

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur
        self.lambda_prior = lambda_prior

        # Disable automatic optimization
        self.automatic_optimization = False

        self.vocoder = BigVGAN(next(self.generator.parameters()).device)


    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            }
        )

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)

        ret = self.generator(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,
                                style_codes,style_code_lengths,durations)

        gen_loss = ret['loss']
        dur_loss = ret['dur_loss']
        prior_loss = ret['prior_loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/dur_loss",
            dur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/prior_loss",
            prior_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss * self.lambda_gen + dur_loss * self.lambda_dur + prior_loss * self.lambda_prior

        # Backward
        optim_g.zero_grad()

        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Manual LR Scheduler
        scheduler_g = self.lr_schedulers()
        scheduler_g.step()

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels, dur_loss, prior_loss = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,
                                style_codes,style_code_lengths,durations)
        ### denormalize
        refine_mels = refine_mels*(self.mel_max - self.mel_min)+ self.mel_min

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]

        refine_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, refine_mels, reduction="none"), mel_masks
        )

        self.log(
            "val/refine_mel_loss",
            refine_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/dur_loss",
            dur_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/prior_loss",
            prior_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.spec2wav(gt_mels)
        gen_audios = self.vocoder.spec2wav(refine_mels)

        for idx, (
            mel,
            refine_mel,
            mel_len,
            gt_audio,
            gen_audio
        ) in enumerate(
            zip(
                gt_mels,
                refine_mels,
                mel_lengths,
                gt_audios.cpu().float(),
                gen_audios.cpu().float()
            )
        ):
            if idx > 4:
                break
            audio_len = mel_len * self.hop_length

            image_mels = plot_mel(
                [
                    refine_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Refine (Flow)",
                    "Ground-Truth",
                ],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        "wavs": [
                            wandb.Audio(
                                gt_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="gt",
                            ),
                            wandb.Audio(
                                gen_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="recon",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    gt_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gen",
                    gen_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)


class FlowDecoder_Wo_Style(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 320,
        sample_rate: int = 24000,
        mel_min: float = -1,
        mel_max: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        
        self.mel_min = mel_min
        self.mel_max = mel_max

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        self.vocoder = BigVGAN(next(self.generator.parameters()).device)


    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            }
        )

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)

        ret = self.generator(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,durations)

        gen_loss = ret['loss']
        dur_loss = ret['dur_loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/dur_loss",
            dur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss * self.lambda_gen + dur_loss * self.lambda_dur

        # Backward
        optim_g.zero_grad()

        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Manual LR Scheduler
        scheduler_g = self.lr_schedulers()
        scheduler_g.step()

    def validation_step(self, batch: Any, batch_idx: int):
        if batch_idx >= 2:
            return
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        durations = batch["durations"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_min) / (self.mel_max - self.mel_min)
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels, dur_loss = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,durations)
        ### denormalize
        refine_mels = refine_mels*(self.mel_max - self.mel_min)+ self.mel_min

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]

        refine_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, refine_mels, reduction="none"), mel_masks
        )

        self.log(
            "val/refine_mel_loss",
            refine_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/dur_loss",
            dur_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.spec2wav(gt_mels)
        gen_audios = self.vocoder.spec2wav(refine_mels)

        for idx, (
            mel,
            refine_mel,
            mel_len,
            gt_audio,
            gen_audio
        ) in enumerate(
            zip(
                gt_mels,
                refine_mels,
                mel_lengths,
                gt_audios.cpu().float(),
                gen_audios.cpu().float()
            )
        ):
            if idx > 4:
                break
            audio_len = mel_len * self.hop_length

            image_mels = plot_mel(
                [
                    refine_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Refine (Flow)",
                    "Ground-Truth",
                ],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        "wavs": [
                            wandb.Audio(
                                gt_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="gt",
                            ),
                            wandb.Audio(
                                gen_audio[:audio_len],
                                sample_rate=self.sampling_rate,
                                caption="recon",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    gt_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gen",
                    gen_audio[:audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)