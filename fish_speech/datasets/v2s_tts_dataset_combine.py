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
from fish_speech.datasets.utils import DistributedDynamicBatchSampler
from fish_speech.datasets.mel_utils import mel_spectrogram
import json
from fish_speech.models.ssl_models.utils.utils import read_wav_16k

from fish_speech.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class FlowTokenDecoderDataset(Dataset):
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        data_json: str = None,
    ):
        super().__init__()

        with open(data_json, 'r') as json_file:
            self.feature_files = json.load(json_file)
        
        self.lengths_list = []
        new_feature_files = []
        for item in self.feature_files:
            code_len = item['code_len']
            if code_len>2000:
                continue
            self.lengths_list.append(code_len)
            new_feature_files.append(item)

        self.feature_files = new_feature_files
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def __len__(self):
        return len(self.feature_files)
    
    def get_mel(self,wav_fp):
        wavform,sr = torchaudio.load(wav_fp)
        if sr!=self.sample_rate:
            wavform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wavform)
            max_val = wavform.abs().max()
            if max_val > 1:
                wavform /= max_val
        mel = mel_spectrogram(wavform)
        return mel[0] # (C,T)
    
    def get_wav(self,wav_fp):
        audio = read_wav_16k(wav_fp)
        audio = torch.from_numpy(audio).float()
        audio = torch.nn.functional.layer_norm(audio, audio.shape)
        return audio


    def get_item(self, idx):
        item = self.feature_files[idx]

        if 'fid' in item:
            code_fp = item['unit_fn']
            audio_file = item['wav_fn']

            spk_embed = torch.zeros((192,)).float()
            ### 根据wav_fn获取code_fn
            hubert_code = np.load(code_fp).astype(int) # [T]
            hubert_code = torch.LongTensor(hubert_code)
            ### hifigan mel
            mel = self.get_mel(audio_file)
            wav = self.get_wav(audio_file)
        else:
            code_fp = item['code_fp']
            audio_file = code_fp.replace('/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/mls_en_hubert_code','/apdcephfs_qy4/share_1213607/shared_data/raw_data/open_source/asr_data/mls_en')
            audio_file = audio_file.replace('.npy','.wav')

            spk_file = code_fp.replace('/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/mls_en_hubert_code','/apdcephfs_qy4/share_1213607/jialongzuo/v100_open/mls_en_spk_embed').split('.')[0]
            spk_file = f'{spk_file}_spk.npy'
            spk_embed = torch.from_numpy(np.load(spk_file))

            ### 根据wav_fn获取code_fn
            hubert_code = np.load(code_fp).astype(int) # [T]
            hubert_code = torch.LongTensor(hubert_code)

            ### hifigan mel
            mel = self.get_mel(audio_file)

            wav = self.get_wav(audio_file)

        return {
            "code": hubert_code,
            "mel": mel,
            "spk_embed":spk_embed,
            "wav":wav
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
class FlowTokenDecoderCollator:

    audio_pad_token: int = 0
    def __call__(self, batch):
        batch = batch[0]
        batch = [x for x in batch if x is not None]

        code_lengths = torch.tensor([x["code"].shape[-1] for x in batch])
        code_maxlen = code_lengths.max()

        mel_lengths = torch.tensor([x["mel"].shape[-1] for x in batch])
        mel_maxlen = mel_lengths.max()

        wav_lengths = torch.tensor([x["wav"].shape[0] for x in batch])
        wav_maxlen = wav_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        codes = []
        spk_embeds = []
        mels = []
        wavs = []
        for x in batch:
            codes.append(
                torch.nn.functional.pad(x["code"], (0, code_maxlen - x["code"].shape[-1]),value=self.audio_pad_token)
            )

            wavs.append(
                torch.nn.functional.pad(x["wav"], (0, wav_maxlen - x["wav"].shape[0]),value=0.0)
            )

            mels.append(
                torch.nn.functional.pad(x["mel"], (0, mel_maxlen - x["mel"].shape[-1]))
            )
            spk_embeds.append(x['spk_embed'])

        return {
            "codes": torch.stack(codes),
            "code_lengths": code_lengths,
            "mels": torch.stack(mels),
            "mel_lengths": mel_lengths,  # (B,C,T)
            "spk_embeds": torch.stack(spk_embeds),
            "wavs": torch.stack(wavs),
            "wav_lengths": wav_lengths,
        }

@dataclass
class FlowTokenDecoderCollator_Val:

    audio_pad_token: int = 0
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        code_lengths = torch.tensor([x["code"].shape[-1] for x in batch])
        code_maxlen = code_lengths.max()

        mel_lengths = torch.tensor([x["mel"].shape[-1] for x in batch])
        mel_maxlen = mel_lengths.max()

        wav_lengths = torch.tensor([x["wav"].shape[0] for x in batch])
        wav_maxlen = wav_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        codes = []
        spk_embeds = []
        mels = []
        wavs = []
        for x in batch:
            codes.append(
                torch.nn.functional.pad(x["code"], (0, code_maxlen - x["code"].shape[-1]),value=self.audio_pad_token)
            )

            wavs.append(
                torch.nn.functional.pad(x["wav"], (0, wav_maxlen - x["wav"].shape[0]),value=0.0)
            )

            mels.append(
                torch.nn.functional.pad(x["mel"], (0, mel_maxlen - x["mel"].shape[-1]))
            )
            spk_embeds.append(x['spk_embed'])

        return {
            "codes": torch.stack(codes),
            "code_lengths": code_lengths,
            "mels": torch.stack(mels),
            "mel_lengths": mel_lengths,  # (B,C,T)
            "spk_embeds": torch.stack(spk_embeds),
            "wavs": torch.stack(wavs),
            "wav_lengths": wav_lengths,
        }


class FlowTokenDecoderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: FlowTokenDecoderDataset,
        val_dataset: FlowTokenDecoderDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
        audio_pad_token: int = 0,
        max_tokens: int = 6400  # Maximum number of tokens per batch
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.audio_pad_token = audio_pad_token

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
            batch_ordering='ascending',
            epoch=0)
        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=FlowTokenDecoderCollator(audio_pad_token=self.audio_pad_token),
            num_workers=self.num_workers//world_size,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=FlowTokenDecoderCollator_Val(audio_pad_token=self.audio_pad_token),
            num_workers=self.num_workers,
            persistent_workers=True,
        )