from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

class DurPredictorTask(L.LightningModule):
    default_monitor: str = "dur_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.val_outputs = []

        # Other parameter

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

        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths)

        dur_loss = ret['dur_loss']
        self.log(
            "train/generator/dur_loss",
            dur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = dur_loss

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

    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        avg_loss = torch.stack([x['dur_loss'] for x in outputs]).mean()
        self.log('dur_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]

        ret = self.generator.inference(mels,mel_lengths,hubert_codes,hubert_code_lengths)
        dur_loss = ret['dur_loss']

        self.log(
            "val/dur_loss",
            dur_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        ret_loss = {}
        ret_loss['dur_loss'] = dur_loss
        self.val_outputs.append(ret_loss)