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

class V2S_TTS_Pretrain(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 22050
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        
        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate

        # Disable automatic optimization
        self.automatic_optimization = False


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

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        spk_embeds = batch['spk_embeds']
        wavs,wav_lengths = batch['wavs'],batch['wav_lengths']

        ret = self.generator(mels,mel_lengths,codes,code_lengths,spk_embeds,wavs,wav_lengths)

        loss = ret['loss']
        self.log(
            "train/generator/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Backward
        optim_g.zero_grad()

        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Manual LR Scheduler
        scheduler_g = self.lr_schedulers()
        scheduler_g.step()

    def validation_step(self, batch: Any, batch_idx: int):
        codes, code_lengths = batch["codes"], batch["code_lengths"]
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        spk_embeds = batch['spk_embeds']
        wavs,wav_lengths = batch['wavs'],batch['wav_lengths']

        gt_mels = mels
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, gt_mels.shape[2]), 1
        ).to(gt_mels.dtype)

        refine_mels = self.generator.inference(gt_mels,mel_lengths,codes,code_lengths,spk_embeds,wavs,wav_lengths)
        ### denormalize

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

        # only log the first batch
        if batch_idx != 0:
            return

        for idx, (
            mel,
            refine_mel,
            mel_len
        ) in enumerate(
            zip(
                gt_mels,
                refine_mels,
                mel_lengths
            )
        ):

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
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )

            plt.close(image_mels)