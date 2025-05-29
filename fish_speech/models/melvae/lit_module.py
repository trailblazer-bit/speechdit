from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
from fish_speech.models.melvae.disc import Discriminator


from fish_speech.models.vqgan.utils import (
    avg_with_mask,
    plot_mel,
    sequence_mask
)

class MelVAE_Task(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        discriminator: nn.Module,
        lambda_mel: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_kl: float = 1.0,
        accumulate_grad_batches: int = 1,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        self.discriminator = discriminator
            
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_kl = lambda_kl
        self.accumulate_grad_batches = accumulate_grad_batches

        # Disable automatic optimization
        self.automatic_optimization = False


    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())
        optimizer_discriminator = self.optimizer_builder(self.discriminator.parameters())

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)
        lr_scheduler_discriminator = self.lr_scheduler_builder(optimizer_discriminator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            },
            {
                "optimizer": optimizer_discriminator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_discriminator,
                    "interval": "step",
                    "name": "optimizer/discriminator",
                },
            }
        )

    def training_step(self, batch, batch_idx):
        optim_g,optim_d = self.optimizers()

        mels, mel_lengths = batch["mels"], batch["mel_lengths"]

        ret = self.generator(mels, mel_lengths)
        loss_kl = ret['kl']
        gen_mel = ret['mel_out']

        # Discriminator
        D_outputs = self.discriminator(mels.transpose(1,2))
        loss_real = 0.5 * torch.mean((D_outputs["y"] - 1) ** 2)
        D_outputs = self.discriminator(gen_mel.detach().transpose(1,2))
        loss_fake = 0.5 * torch.mean(D_outputs["y"] ** 2)

        loss_d = loss_real + loss_fake

        self.log(
            "train/discriminator/loss",
            loss_d,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Discriminator backward
        # optim_d.zero_grad()
        self.manual_backward(loss_d)
        self.clip_gradients(
            optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        if (batch_idx+1)%self.accumulate_grad_batches==0:
            optim_d.step()
            optim_d.zero_grad()


        ### loss_mel
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)

        min_mel_length = min(mels.shape[-1], gen_mel.shape[-1])
        mels = mels[:, :, :min_mel_length]
        gen_mel = gen_mel[:, :, :min_mel_length]

        loss_mel = avg_with_mask(
            F.l1_loss(mels, gen_mel, reduction="none"), mel_masks
        )

        # Adversarial Loss
        loss_adv = 0.5 * torch.mean((self.discriminator(gen_mel.transpose(1,2))["y"] - 1) ** 2)

        # Total loss
        loss = (
            self.lambda_mel * loss_mel
            + self.lambda_adv * loss_adv
            + self.lambda_kl * loss_kl
        )

        # Log losses
        self.log(
            "train/generator/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/generator/loss_mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/generator/loss_kl",
            loss_kl,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/generator/loss_adv",
            loss_adv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # Generator backward
        # optim_g.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        if (batch_idx+1)%self.accumulate_grad_batches==0:
            optim_g.step()
            optim_g.zero_grad()

        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        with torch.no_grad():
            gt_mels = mels
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, gt_mels.shape[2]), 1
        ).to(gt_mels.dtype)

        ret = self.generator.inference(gt_mels, mel_lengths)
        refine_mels = ret['mel_out']

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]

        refine_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, refine_mels, reduction="none"), mel_masks
        )

        loss_kl = ret['kl']
        self.log(
            "val/recon_mel_loss",
            refine_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/kl_loss",
            loss_kl,
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