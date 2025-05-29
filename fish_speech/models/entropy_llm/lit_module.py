from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn


class Entropy_LLM_Task(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        vocab_size: int =0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        # Disable automatic optimization
        self.automatic_optimization = False

        self.vocab_size = vocab_size

    
    def configure_optimizers(self):
        # Get weight decay parameters
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        # for i in optimizer.param_groups:
        #     log.info(
        #         f"Set weight decay: {i['weight_decay']} for {len(i['params'])} parameters"
        #     )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def get_accuracy(self, logits, labels):
        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[labels == self.vocab_size-1] = 0
        correct = correct.sum()
        accuracy = correct / (labels != (self.vocab_size-1)).sum()

        return accuracy

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        

        loss = self.generator(codes,labels)

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
        labels, label_lengths = batch["labels"], batch["label_lengths"]

        with torch.no_grad():
            logits = self.generator(codes)

            base_loss = F.nll_loss(
                F.log_softmax(logits.flatten(end_dim=-2).float(), -1),
                labels.flatten(end_dim=-1),
                ignore_index=self.vocab_size-1)
            
            self.log(
                f"val/loss",
                base_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            # Top-5 accuracy
            accuracy = self.get_accuracy(logits, labels)
            self.log(
                f"val/top_5_accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )