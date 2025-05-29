from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from fish_speech.models.maskdur_predictor.mask import make_pad_mask
from matplotlib import pyplot as plt
from torch import nn

class DurPredictorTask(L.LightningModule):
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

        dur_tokens = batch["dur_tokens"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        spk_embeds = batch['spk_embeds']
        dur_mask = (~make_pad_mask(hubert_code_lengths)).to(spk_embeds)  # (B,T)

        logits, final_mask, x0, _, _ = self.generator(x0=dur_tokens,x_mask=dur_mask,content_token=hubert_codes,content_mask=dur_mask,spk_embed=spk_embeds)

        dur_loss = self.compute_loss(logits,x0,final_mask)

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

    def validation_step(self, batch: Any, batch_idx: int):
        dur_tokens = batch["dur_tokens"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        spk_embeds = batch['spk_embeds']
        dur_mask = (~make_pad_mask(hubert_code_lengths)).to(spk_embeds)  # (B,T)

        T = dur_tokens.shape[-1]
        prompt = dur_tokens[:,:T//2]
        gt_dur_token = dur_tokens[:,T//2:]
        target_len = T - T//2
        content_token = hubert_codes
        content_mask = dur_mask

        pred_dur_token = self.generator.reverse_diffusion(prompt,target_len,content_token,content_mask,\
            spk_embeds,n_timesteps=25,cfg=2.5,rescale_cfg=0.75)


        # pred_dur_token = pred_dur_token[:, :target_len]  
        # gt_dur_token = gt_dur_token[:, :target_len] 

        # pred_dur_token 和 gt_dur_token 都是离散的分类标签
        correct_predictions = (pred_dur_token == gt_dur_token).float() 
        accuracy = correct_predictions.sum() / (target_len*dur_tokens.shape[0])
        mean_accuracy = accuracy.mean() 

        self.log(
            "val/dur_accuracy",
            mean_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )