import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import os
import torchaudio
sys.path.append('')

import librosa
import numpy as np
import torch
import json
from collections import defaultdict
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
from fish_speech.datasets.utisl import DistributedDynamicBatchSampler

from fish_speech.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class FlowDecoderDataset(Dataset):
    def __init__(
        self,
        data_json: str,
        sample_rate: int = 22050,
        hop_length: int = 256
    ):
        super().__init__()

        with open(data_json, 'r') as json_file:
            self.feature_files = json.load(json_file)
        
        self.lengths_list = []
        for item in self.feature_files:
            code_len = item['code_len']
            self.lengths_list.append(code_len)

        # with open(lens_json, 'r') as json_file:
        #     self.lengths_list = json.load(json_file)
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def __len__(self):
        return len(self.feature_files)

    def deduplication(self,hubert_code):
        ### 去重后的hubert code
        hubert_code, dur = torch.unique_consecutive(hubert_code, return_counts=True)
        return hubert_code, dur

    def get_item(self, idx):
        file_map = self.feature_files[idx]
        hubert_fp = file_map['code_fp']
        spk_fp = file_map['spk_fp']

        hubert_code = np.load(hubert_fp)
        hubert_code = torch.from_numpy(hubert_code)
        ## semantic code后去重
        hubert_code, dur = self.deduplication(hubert_code)

        spk_embed = torch.from_numpy(np.load(spk_fp))

        return {
            "dur": dur,
            "hubert_code": hubert_code,
            "spk_embed": spk_embed
        }
    
    def get_item_exception(self,idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error loading {self.feature_files[idx]}: {e}")
            return None

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.get_item_exception(i) for i in idx]
            return batch
        else:
            return self.get_item_exception(idx)

@dataclass
class FlowDecoderCollator:
    semantic_vocab: int = 500,
    def __call__(self, batch):
        batch = batch[0]
        batch = [x for x in batch if x is not None]
        hubert_code_lengths = torch.tensor([len(x["hubert_code"]) for x in batch])
        hubert_code_maxlen = hubert_code_lengths.max()

        dur_lengths = torch.tensor([len(x["dur"]) for x in batch])
        dur_maxlen = dur_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        hubert_codes = []
        dur_tokens = []
        spk_embeds = []

        for x in batch:
            hubert_codes.append(
                torch.nn.functional.pad(x["hubert_code"], (0, hubert_code_maxlen - len(x["hubert_code"])),value=self.semantic_vocab)
            )

            dur_tokens.append(
                torch.nn.functional.pad(x["dur"], (0, dur_maxlen - len(x["dur"])),value=0)
            )
            spk_embeds.append(x['spk_embed'])

        return {
            "dur_tokens": torch.stack(dur_tokens),
            "hubert_codes": torch.stack(hubert_codes),
            "hubert_code_lengths": hubert_code_lengths,
            "spk_embeds": torch.stack(spk_embeds)
        }

@dataclass
class FlowDecoderCollator_Val:
    semantic_vocab: int = 500,
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        hubert_code_lengths = torch.tensor([len(x["hubert_code"]) for x in batch])
        hubert_code_maxlen = hubert_code_lengths.max()

        dur_lengths = torch.tensor([len(x["dur"]) for x in batch])
        dur_maxlen = dur_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        hubert_codes = []
        dur_tokens = []
        spk_embeds = []

        for x in batch:
            hubert_codes.append(
                torch.nn.functional.pad(x["hubert_code"], (0, hubert_code_maxlen - len(x["hubert_code"])),value=self.semantic_vocab)
            )

            dur_tokens.append(
                torch.nn.functional.pad(x["dur"], (0, dur_maxlen - len(x["dur"])),value=0)
            )
            spk_embeds.append(x['spk_embed'])

        return {
            "dur_tokens": torch.stack(dur_tokens),
            "hubert_codes": torch.stack(hubert_codes),
            "hubert_code_lengths": hubert_code_lengths,
            "spk_embeds": torch.stack(spk_embeds)
        }

class FlowDecoderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: FlowDecoderDataset,
        val_dataset: FlowDecoderDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
        semantic_vocab: int = 500,
        max_tokens: int = 9600  # Maximum number of tokens per batch
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.semantic_vocab = semantic_vocab
        self.max_tokens = max_tokens

        self.args = {'max_num_tokens':max_tokens,'num_buckets':6,'audio_max_length':40,'encodec_sr':50,'seed':1234}


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
            collate_fn=FlowDecoderCollator(semantic_vocab=self.semantic_vocab),
            num_workers=self.num_workers//world_size,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=FlowDecoderCollator_Val(semantic_vocab=self.semantic_vocab),
            num_workers=self.num_workers,
            persistent_workers=True,
        )