from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from fish_speech.models.maskdur_predictor.mask import make_pad_mask
from matplotlib import pyplot as plt
from torch import nn

class V2SUnitPredictorTask(L.LightningModule):
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

    def compute_loss(self,logits, x0, final_mask):
        # logits: (B, T, codebook_size) -> (B*T, codebook_size)
        # x0: (B, T) -> (B*T)
        B, T, codebook_size = logits.shape
        logits = logits.view(B * T, codebook_size)
        x0 = x0.view(B * T)
        final_mask = final_mask.view(B * T)

        loss = F.cross_entropy(logits, x0, reduction='none')

        loss = loss * final_mask  

        valid_count = final_mask.sum()
        if valid_count > 0:
            loss = loss.sum() / valid_count
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(logits.device)

        return loss

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        video_features, video_feature_lengths = batch["video_features"], batch["video_feature_lengths"]

        code_mask = (~make_pad_mask(code_lengths)).to(video_features)  # (B,T)

        logits, final_mask, x0, _, _ = self.generator(x0=codes,x_mask=code_mask,video_features=video_features, video_feature_lengths=video_feature_lengths)

        ce_loss = self.compute_loss(logits,x0,final_mask)

        self.log(
            "train/generator/ce_loss",
            ce_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = ce_loss

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
        video_features, video_feature_lengths = batch["video_features"], batch["video_feature_lengths"]
        code_mask = (~make_pad_mask(code_lengths)).to(video_features)  # (B,T)

        target_len = codes.shape[1]

        pred_token = self.generator.reverse_diffusion(target_len,video_features,video_feature_lengths,n_timesteps=25,cfg=2.5,rescale_cfg=0.75)


        # pred_dur_token = pred_dur_token[:, :target_len]  
        # gt_dur_token = gt_dur_token[:, :target_len] 

        # pred_dur_token 和 gt_dur_token 都是离散的分类标签
        correct_predictions = ((pred_token == codes)*code_mask).float() 
        mean_accuracy = correct_predictions.sum() / (torch.sum(mask))

        self.log(
            "val/pred_accuracy",
            mean_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )