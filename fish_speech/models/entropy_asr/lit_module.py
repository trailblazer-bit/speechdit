from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
from transformers import AutoProcessor
from fish_speech.models.entropy_llm.llm import LMTransformer
from fish_speech.models.entropy_asr.utils import entropy,patch_ids_from_lengths,patch_reduce
from infer.patcher import PatcherArgs,Patcher
from transformers import WhisperProcessor
from infer.metric.wer import get_wer_params,calculate_wer


class Entropy_ASR_Task(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        # Disable automatic optimization
        self.automatic_optimization = False

        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    
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

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        

        ret = self.generator(codes,code_lengths,labels,label_lengths,texts,text_lengths)
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
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        with torch.no_grad():
            pred_tokens = self.generator.inference(codes,code_lengths,text_bos_token=self.processor.tokenizer.bos_token_id)

            total_correct = 0
            total_tokens = 0

            for pred, label, label_length in zip(pred_tokens, labels, label_lengths):
                effective_length = min(len(pred), label_length.item())

                correct = (pred[:effective_length] == label[:effective_length]).sum().item()
                total_correct += correct
                total_tokens += effective_length

            accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

            self.log(
                f"val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )


class Entropy_ASR__Whisper_Tokenizer_Task(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        # Disable automatic optimization
        self.automatic_optimization = False

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    
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

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        

        ret = self.generator(codes,code_lengths,labels,label_lengths,texts,text_lengths)
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
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        with torch.no_grad():
            pred_tokens = self.generator.inference(codes,code_lengths,labels,label_lengths,texts,text_lengths)

            total_correct = 0
            total_tokens = 0

            for pred, label, label_length in zip(pred_tokens, labels, label_lengths):
                effective_length = min(len(pred), label_length.item())

                correct = (pred[:effective_length] == label[:effective_length]).sum().item()
                total_correct += correct
                total_tokens += effective_length

            accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

            self.log(
                f"val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )



class Patcher_ASR_Task(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        entropy_llm_ckpt: str = None,
        entropy_llm_config = None,
        patcher_args = None,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        checkpoint = torch.load(entropy_llm_ckpt)
        state_dicts = checkpoint['state_dict']
        new_state_dicts = {}
        for k,v in state_dicts.items():
            new_k = k[10:]
            new_state_dicts[new_k] = v
        self.entropy_llm = LMTransformer(entropy_llm_config)
        self.entropy_llm.load_state_dict(new_state_dicts,strict=True)
        self.entropy_llm.eval()
        for param in self.entropy_llm.parameters():
            param.requires_grad = False

        self.patcher = Patcher(patcher_args)


        # Disable automatic optimization
        self.automatic_optimization = False

        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    
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

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
    
    def prepare_patch_ids(self,codes,code_lengths):
        seq_len = codes.shape[-1]
        with torch.no_grad():
            logits = self.entropy_llm(codes)
            pred_entropies = entropy(logits)
            patch_lengths, scores = self.patcher.patch(tokens=codes,include_next_token=False,entropies=pred_entropies)
            patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)
            num_patches = patch_lengths.shape[1]
            return patch_ids,num_patches

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)
        
        ret = self.generator(codes,code_lengths,labels,label_lengths,texts,text_lengths,patch_ids,num_patches)
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
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)

        with torch.no_grad():
            pred_tokens = self.generator.inference(codes,code_lengths,patch_ids,num_patches,\
                text_bos_token=self.processor.tokenizer.bos_token_id)

            total_correct = 0
            total_tokens = 0

            for pred, label, label_length in zip(pred_tokens, labels, label_lengths):
                effective_length = min(len(pred), label_length.item())

                correct = (pred[:effective_length] == label[:effective_length]).sum().item()
                total_correct += correct
                total_tokens += effective_length

            accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

            self.log(
                f"val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )



class Patcher_ASR_Task_Whisper_Tokenizer(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        entropy_llm_ckpt: str = None,
        entropy_llm_config = None,
        patcher_args = None,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        checkpoint = torch.load(entropy_llm_ckpt)
        state_dicts = checkpoint['state_dict']
        new_state_dicts = {}
        for k,v in state_dicts.items():
            new_k = k[10:]
            new_state_dicts[new_k] = v
        self.entropy_llm = LMTransformer(entropy_llm_config)
        self.entropy_llm.load_state_dict(new_state_dicts,strict=True)
        self.entropy_llm.eval()
        for param in self.entropy_llm.parameters():
            param.requires_grad = False

        self.patcher = Patcher(patcher_args)


        # Disable automatic optimization
        self.automatic_optimization = False

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    
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

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
    
    def prepare_patch_ids(self,codes,code_lengths):
        seq_len = codes.shape[-1]
        with torch.no_grad():
            logits = self.entropy_llm(codes)
            pred_entropies = entropy(logits)
            patch_lengths, scores = self.patcher.patch(tokens=codes,include_next_token=False,entropies=pred_entropies)
            patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)
            num_patches = patch_lengths.shape[1]
            return patch_ids,num_patches

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)
        
        ret = self.generator(codes,code_lengths,labels,label_lengths,texts,text_lengths,patch_ids,num_patches)
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
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)

        with torch.no_grad():
            pred_tokens = self.generator.inference(codes,code_lengths,patch_ids,num_patches,\
                text_bos_token=self.processor.tokenizer.bos_token_id)

            total_correct = 0
            total_tokens = 0

            for pred, label, label_length in zip(pred_tokens, labels, label_lengths):
                effective_length = min(len(pred), label_length.item())

                correct = (pred[:effective_length] == label[:effective_length]).sum().item()
                total_correct += correct
                total_tokens += effective_length

            accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

            self.log(
                f"val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )


class Patcher_ASR_Task_Whisper_Tokenizer_WER(L.LightningModule):
    default_monitor: str = "wer"
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        entropy_llm_ckpt: str = None,
        entropy_llm_config = None,
        patcher_args = None,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator
        self.generator = generator

        checkpoint = torch.load(entropy_llm_ckpt)
        state_dicts = checkpoint['state_dict']
        new_state_dicts = {}
        for k,v in state_dicts.items():
            new_k = k[10:]
            new_state_dicts[new_k] = v
        self.entropy_llm = LMTransformer(entropy_llm_config)
        self.entropy_llm.load_state_dict(new_state_dicts,strict=True)
        self.entropy_llm.eval()
        for param in self.entropy_llm.parameters():
            param.requires_grad = False

        self.patcher = Patcher(patcher_args)


        # Disable automatic optimization
        self.automatic_optimization = False

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        self.val_outputs = []

    
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

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
    
    def prepare_patch_ids(self,codes,code_lengths):
        seq_len = codes.shape[-1]
        with torch.no_grad():
            logits = self.entropy_llm(codes)
            pred_entropies = entropy(logits)
            patch_lengths, scores = self.patcher.patch(tokens=codes,include_next_token=False,entropies=pred_entropies)
            patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)
            num_patches = patch_lengths.shape[1]
            return patch_ids,num_patches

    def training_step(self, batch, batch_idx):
        optim_g = self.optimizers()

        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)
        
        ret = self.generator(codes,code_lengths,labels,label_lengths,texts,text_lengths,patch_ids,num_patches)
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
    
    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        avg_loss = torch.stack([x['wer'] for x in outputs]).mean()
        self.log('wer', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def validation_step(self, batch: Any, batch_idx: int):
        codes, code_lengths = batch["codes"], batch["code_lengths"]
        labels, label_lengths = batch["labels"], batch["label_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]
        raw_text = batch["raw_text"]

        patch_ids,num_patches = self.prepare_patch_ids(codes,code_lengths)

        with torch.no_grad():
            pred_tokens = self.generator.inference_topk_topp(codes,code_lengths,patch_ids,\
                num_patches,text_bos_token=self.processor.tokenizer.bos_token_id)
            if (pred_tokens[:,0]==50257).all():
                pred_tokens = pred_tokens[:,1:]
            if (pred_tokens[:,-1]==50258).all():
                pred_tokens = pred_tokens[:,:-1]
            transcription = self.processor.batch_decode(pred_tokens)
            wer = get_wer_params(transcription[0].lower(),raw_text[0])
            wer = torch.tensor(wer).float().to(codes.device)

            self.log(
                f"val/wer",
                wer,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            ret_loss = {}
            ret_loss['wer'] = wer
            self.val_outputs.append(ret_loss)
        