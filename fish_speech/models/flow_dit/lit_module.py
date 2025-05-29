from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
# from fish_speech.models.flow_vqgan.modules.discriminator import Discriminator
from fish_speech.models.stabletts.modules.dscrm import Discriminator


from fish_speech.models.vqgan.utils import (
    avg_with_mask,
    plot_mel,
    sequence_mask
)
from fish_speech.third_party.vocoder_infer import BigVGAN

class FlowDit_Wi_Style(L.LightningModule):
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
        min_level_db: float = -120,
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
        self.min_level_db = min_level_db

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels * 20 - self.min_level_db) / (15 - self.min_level_db)

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels * 20 - self.min_level_db) / (15 - self.min_level_db)
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths)
        ### denormalize
        refine_mels = (refine_mels*(15 - self.min_level_db)+ self.min_level_db)/20

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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


class FlowDit_Wi_Style_Mean_Norm(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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


class FlowDit_Wi_Style_Mean_Norm_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.val_outputs = []
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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

        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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

class FlowDit_Wi_Style_Prepend_Mean_Norm_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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

class FlowDit_Wi_Style_Prepend_HifiMel_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from fish_speech.models.hifigan.generator import HiFTGenerator
        from fish_speech.models.hifigan.f0_predictor import ConvRNNF0Predictor
        f0_net = ConvRNNF0Predictor()
        self.vocoder = HiFTGenerator(f0_predictor=f0_net)
        checkpoint = torch.load('/workspace/user_code/kuachen/projects/fish-speech/checkpoints/hift.pt')
        self.vocoder.load_state_dict(checkpoint)
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)
        gt_mels = mels

        refine_mels = self.generator.inference(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.inference(gt_mels)[0]
        gen_audios = self.vocoder.inference(refine_mels)[0]

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

class FlowDit_Wi_MelMaskCond_HifiMel_DBatch_Wi_F0(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from fish_speech.models.hifigan.generator import HiFTGenerator
        from fish_speech.models.hifigan.f0_predictor import ConvRNNF0Predictor
        f0_net = ConvRNNF0Predictor()
        self.vocoder = HiFTGenerator(f0_predictor=f0_net)
        checkpoint = torch.load('/workspace/user_code/kuachen/projects/fish-speech/checkpoints/hift.pt')
        self.vocoder.load_state_dict(checkpoint)
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)
        gt_mels = mels

        refine_mels = self.generator.inference(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.inference(gt_mels)[0]
        gen_audios = self.vocoder.inference(refine_mels)[0]

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

class FlowDit_Wi_Style_Add_HifiMel_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from fish_speech.models.hifigan.generator import HiFTGenerator
        from fish_speech.models.hifigan.f0_predictor import ConvRNNF0Predictor
        f0_net = ConvRNNF0Predictor()
        self.vocoder = HiFTGenerator(f0_predictor=f0_net)
        checkpoint = torch.load('/workspace/user_code/kuachen/projects/fish-speech/checkpoints/hift.pt')
        self.vocoder.load_state_dict(checkpoint)
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)
        gt_mels = mels

        refine_mels = self.generator.inference(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.inference(gt_mels)[0]
        gen_audios = self.vocoder.inference(refine_mels)[0]

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

class FlowDit_Wi_MelMaskCond_HifiMel_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from fish_speech.models.hifigan.generator import HiFTGenerator
        from fish_speech.models.hifigan.f0_predictor import ConvRNNF0Predictor
        f0_net = ConvRNNF0Predictor()
        self.vocoder = HiFTGenerator(f0_predictor=f0_net)
        checkpoint = torch.load('/workspace/user_code/kuachen/projects/fish-speech/checkpoints/hift.pt')
        self.vocoder.load_state_dict(checkpoint)
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)
        gt_mels = mels

        refine_mels = self.generator.inference(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds)

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.inference(gt_mels)[0]
        gen_audios = self.vocoder.inference(refine_mels)[0]

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

class FlowDit_Wi_HubertOnly_HifiMel_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from fish_speech.models.hifigan.generator import HiFTGenerator
        from fish_speech.models.hifigan.f0_predictor import ConvRNNF0Predictor
        f0_net = ConvRNNF0Predictor()
        self.vocoder = HiFTGenerator(f0_predictor=f0_net)
        checkpoint = torch.load('/workspace/user_code/kuachen/projects/fish-speech/checkpoints/hift.pt')
        self.vocoder.load_state_dict(checkpoint)
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        ret = self.generator(mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, mels.shape[2]), 1
        ).to(mels.dtype)
        gt_mels = mels

        refine_mels = self.generator.inference(gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,spk_embeds)

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.inference(gt_mels)[0]
        gen_audios = self.vocoder.inference(refine_mels)[0]

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

class FlowDit_Wi_Style_Add_Mean_Norm_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        self.val_outputs = []

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        style_codes, style_code_lengths = batch["style_codes"], batch["style_code_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,style_codes,style_code_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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
        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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

class FlowDit_Wi_Text_Mean_Norm_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.val_outputs = []
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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

        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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


class FlowDit_Cross_Text_Mean_Norm_DBatch(L.LightningModule):
    default_monitor: str = "refine_mel_loss"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        hop_length: int = 256,
        sample_rate: int = 24000,
        mel_mean: float = -1,
        mel_std: float = 1,
        lambda_gen: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator
        
        self.val_outputs = []
        
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.lambda_gen =lambda_gen
        self.lambda_dur = lambda_dur

        # Disable automatic optimization
        self.automatic_optimization = False

        from vocos import Vocos
        self.vocoder = Vocos.from_pretrained("/workspace/user_code/kuachen/projects/vocos/pretrained/pytorch_model.bin",\
            '/workspace/user_code/kuachen/projects/vocos/pretrained/config.yaml')
        for p in self.vocoder.parameters():
            p.requires_grad = False


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
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std

        ret = self.generator(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds)

        gen_loss = ret['loss']
        self.log(
            "train/generator/loss",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = gen_loss

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
        avg_loss = torch.stack([x['refine_mel_loss'] for x in outputs]).mean()
        self.log('refine_mel_loss', avg_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        mels, mel_lengths = batch["mels"], batch["mel_lengths"]
        hubert_codes, hubert_code_lengths = batch["hubert_codes"], batch["hubert_code_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        spk_embeds = batch["spk_embeds"]

        with torch.no_grad():
            gt_mels = mels
            ### normalize
            norm_gt_mels = (gt_mels - self.mel_mean) / self.mel_std
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, norm_gt_mels.shape[2]), 1
        ).to(norm_gt_mels.dtype)

        refine_mels = self.generator.inference(norm_gt_mels,mel_lengths,hubert_codes,hubert_code_lengths,texts,text_lengths,spk_embeds)
        ### denormalize
        refine_mels = refine_mels*self.mel_std + self.mel_mean

        min_mel_length = min(gt_mels.shape[-1], refine_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        refine_mels = refine_mels[:, :, :min_mel_length]
        mel_masks = mel_masks[:, :, :min_mel_length]

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

        ret_loss = {}
        ret_loss['refine_mel_loss'] = refine_mel_loss
        self.val_outputs.append(ret_loss)

        # only log the first batch
        if batch_idx != 0:
            return

        gt_audios = self.vocoder.decode(gt_mels)
        gen_audios = self.vocoder.decode(refine_mels)

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