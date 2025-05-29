import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import os
from tqdm import tqdm
sys.path.append('')

import librosa
import numpy as np
import torch
import torchaudio
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
from fish_speech.datasets.utisl import DistributedDynamicBatchSampler
from fish_speech.datasets.mel_utils import mel_spectrogram
import json

from fish_speech.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

class SpeechDitDataset(Dataset):
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        split = 'train',
        save_items_fp = "",
        save_lens_fp = "",
        stride = 4,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.stride = stride
        self.split = split

        with open(save_items_fp,'r') as fp:
            self.data_items = json.load(fp)
        with open(save_lens_fp,'r') as fp:
            self.lengths_list = json.load(fp)
        
        if split == 'valid':
            self.data_items = self.data_items[:200]
            self.lengths_list = self.lengths_list[:200]

    def __len__(self):
        return len(self.data_items)
    
    def get_mel(self,wav_fn):
        wavform,sr = torchaudio.load(wav_fn)
        resampler = torchaudio.transforms.Resample(orig_freq=sr,new_freq=self.sample_rate)
        resampled_wavform = resampler(wavform)
        max_val = resampled_wavform.abs().max()
        if max_val>1:
            resampled_wavform/=max_val
        mel = mel_spectrogram(resampled_wavform)
        return mel[0]

    def get_item(self, idx):
        item = self.data_items[idx]
        audio_file = item['wav_fp']
        if 'EN' in audio_file:
            audio_file = f'/nfs1/ggyzhang/dataset/Emilia/{audio_file}'
        else:
            audio_file = f'/nfs1/ggy/projects/data/{audio_file}'

        ### hifigan mel
        mel = self.get_mel(audio_file)

        C,T = mel.shape
        if T%self.stride!=0:
            new_T = T - (T%self.stride)
            mel = mel[:,:new_T]
            
        return {
            "mel": mel
        }

    def get_item_exception(self,idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error loading {self.data_items[idx]}: {e}")
            return None

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.get_item_exception(i) for i in idx]
            return batch
        else:
            return self.get_item_exception(idx)


@dataclass
class FlowTokenDecoderCollator:

    def __call__(self, batch):
        batch = batch[0]
        batch = [x for x in batch if x is not None]

        mel_lengths = torch.tensor([x["mel"].shape[-1] for x in batch])
        mel_maxlen = mel_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        mels = []
        for x in batch:
            mels.append(
                torch.nn.functional.pad(x["mel"], (0, mel_maxlen - x["mel"].shape[-1]))
            )

        return {
            "mels": torch.stack(mels),
            "mel_lengths": mel_lengths,
        }

@dataclass
class FlowTokenDecoderCollator_Val:

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        mel_lengths = torch.tensor([x["mel"].shape[-1] for x in batch])
        mel_maxlen = mel_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        mels = []
        for x in batch:
            mels.append(
                torch.nn.functional.pad(x["mel"], (0, mel_maxlen - x["mel"].shape[-1]))
            )

        return {
            "mels": torch.stack(mels),
            "mel_lengths": mel_lengths,
        }


class FlowTokenDecoderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: SpeechDitDataset,
        val_dataset: SpeechDitDataset,
        batch_size: int = 32,
        num_workers: int = 8,
        val_batch_size: Optional[int] = None,
        max_tokens: int = 9600  # Maximum number of tokens per batch
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.max_tokens = max_tokens

        self.args = {'max_num_tokens':max_tokens,'num_buckets':6,'audio_max_length':35,'encodec_sr':87,'seed':1234}


    def train_dataloader(self):
        world_size = torch.distributed.get_world_size()
        train_sampler = DistributedDynamicBatchSampler(
            self.train_dataset, 
            self.args, 
            num_replicas=world_size, 
            rank=torch.distributed.get_rank(), 
            shuffle=True, 
            seed=self.args['seed'], 
            drop_last=True, 
            lengths_list=self.train_dataset.lengths_list, 
            verbose=True, 
            epoch=0)
        
        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=FlowTokenDecoderCollator(),
            num_workers=self.num_workers//world_size,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=FlowTokenDecoderCollator_Val(),
            num_workers=self.num_workers,
            persistent_workers=True,
        )