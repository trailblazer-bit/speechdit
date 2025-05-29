# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
import json
from typing import Any, List, Optional, Union

sys.path.insert(0,'/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/fairseq/fairseq')

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile

from dataclasses import dataclass
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
from fish_speech.datasets.utils import DistributedDynamicBatchSampler

from fish_speech.datasets import custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep):

    with open(manifest_path, 'r') as json_file:
        feature_files = json.load(json_file)

    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    for ind,item in enumerate(feature_files):
        sz = int(item['v_frames'])
        if min_keep is not None and sz < min_keep:
            n_short += 1
        elif max_keep is not None and sz > max_keep:
            n_long += 1
        else:
            names.append(item)
            inds.append(ind)
            sizes.append(sz)
    tot = len(feature_files)
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long"
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return names, inds, tot, sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class AVHubertDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1
    ):
        self.label_rates = [-1]
        self.modalities = set(modalities)
        self.names, inds, tot, self.sizes = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.single_target = single_target
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num
        # self.lang_dict = {'en':"English", 'es':"Spanish", 'fr':"French", 'it':'Italian', 'pt':"Portuguese"}

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )


    def load_units(self, index):
        #assert('video' in self.modalities and 'audio' in self.modalities)
        av_units = self.cluster_counts[index].strip().split(' ')
        int_av_units = [int(x) for x in av_units]
        av_units = torch.tensor(int_av_units, dtype=int)

        return av_units
        
    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            audio_npy = audio_fn.replace('.wav','.npy')
            if os.path.exists(audio_npy) == False:
                sample_rate, wav_data = wavfile.read(audio_fn)
                assert sample_rate == 16_000 and len(wav_data.shape) == 1
                audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
                audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
                np.save(audio_npy[:-4], audio_feats)
            else:
                audio_feats = np.load(audio_npy)
                
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed


    def get_item(self, index):
        item = self.names[index]
        item_name = item['fid']
        audio_fn = item['wav_fn']
        video_fn = item['video_fn']
        unit_fn = item['unit_fn']

        video_feats, audio_feats = self.load_feature((video_fn,None))
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        tokens = torch.from_numpy(np.load(unit_fn)).astype(int)
        return {"id": index, 'item_name': item_name, "video_source": video_feats,'audio_source': audio_feats,"code":tokens}
    
    def get_item_exception(self,idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error loading {self.names[idx]}: {e}")
            return None

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.get_item_exception(i) for i in idx]
            return batch
        else:
            return self.get_item_exception(idx)

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        ############# cluster_counts ############
        cluster_counts_source = [s["cluster_counts"] for s in samples]

        ############# av_hubert ############
        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        
        text_instructions = [
            [s["text_source"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        

        collated_texts,_ ,_  = self.collater_label(
           text_instructions, audio_size, audio_starts
        )
        
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )
        text_attn_mask = collated_texts[0][0] != 0
        target_attn_mask = targets_list[0][0] != 0

        source = {"audio": collated_audios, "video": collated_videos, "cluster_counts": cluster_counts_source, "text": collated_texts[0][0]}

        net_input = {"source": source, "padding_mask": padding_mask, 'text_attn_mask': text_attn_mask}

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples]
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            if self.is_s2s:
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
                batch['target_attn_mask'] = target_attn_mask
            else:
                batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch
        
        
    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens


    def collater_seq_label_llm(self, targets):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = 0, self.llm_tokenizer.eos_token_id
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
       
        new_targets = []
        for tar in targets:
            new_targets.append(tar[1:])

        prev_output_tokens = data_utils.collate_tokens(new_targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=False)
        
        
        prev_list = []
        for prev_tokens in prev_output_tokens:
            padding_start_idx = torch.sum(prev_tokens == 0) * -1
            if padding_start_idx == 0:
                prev_list.append(torch.cat((prev_tokens, torch.tensor([2]).long())))
            else:
                prev_tokens[padding_start_idx] = 2
                prev_list.append(torch.cat((prev_tokens, torch.tensor([0]).long())))
        
        prev_output_tokens = torch.stack(prev_list, dim=0)
        return (targets_, prev_output_tokens), lengths, ntokens


    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates)
        for targets, label_rate in itr:
            if label_rate == -1:  
                targets, lengths, ntokens = self.collater_seq_label_llm(targets)
            else:
                raise NotImplementedError("not yet")
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list


    def collate_tokens(self,
        values,
        pad_idx,
        eos_idxs,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst, eos_idx):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)], eos_idxs[i])
        return res



    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]



@dataclass
class FlowDecoderCollator:
    audio_pad_token: int = 0
    def __call__(self, batch):
        batch = batch[0]
        batch = [x for x in batch if x is not None]

        code_lengths = torch.tensor([x["code"].shape[-1] for x in batch])
        code_maxlen = code_lengths.max()

        video_feature_lengths = torch.tensor([x["video_source"].shape[0] for x in batch])
        video_feature_maxlen = video_feature_lengths.max()

        video_features = []
        codes = []
        for x in batch:
            codes.append(
                torch.nn.functional.pad(x["code"], (0, code_maxlen - x["code"].shape[-1]),value=self.audio_pad_token)
            )
            v_feature = x["video_source"]
            t, h, w, c = v_feature.shape
            padding = (0, 0, 0, 0, 0, 0, 0, video_feature_maxlen - t)
            padded_v_feature = torch.nn.functional.pad(v_feature, padding, mode='constant', value=0)
            video_features.append(padded_v_feature)

        return {
            "codes": torch.stack(codes),
            "code_lengths": code_lengths,
            "video_features": torch.stack(video_features),
            "video_feature_lengths": video_feature_lengths
        }

@dataclass
class FlowDecoderCollator_Val:
    audio_pad_token: int = 0
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        code_lengths = torch.tensor([x["code"].shape[-1] for x in batch])
        code_maxlen = code_lengths.max()

        video_feature_lengths = torch.tensor([x["video_source"].shape[0] for x in batch])
        video_feature_maxlen = video_feature_lengths.max()

        video_features = []
        codes = []
        for x in batch:
            codes.append(
                torch.nn.functional.pad(x["code"], (0, code_maxlen - x["code"].shape[-1]),value=self.audio_pad_token)
            )
            v_feature = x["video_source"]
            t, h, w, c = v_feature.shape
            padding = (0, 0, 0, 0, 0, 0, 0, video_feature_maxlen - t)
            padded_v_feature = torch.nn.functional.pad(v_feature, padding, mode='constant', value=0)
            video_features.append(padded_v_feature)

        return {
            "codes": torch.stack(codes),
            "code_lengths": code_lengths,
            "video_features": torch.stack(video_features), ## (B,T,H,W,C)
            "video_feature_lengths": video_feature_lengths 
        }

class FlowDecoderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: AVHubertDataset,
        val_dataset: AVHubertDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
        token_vocab: int = 8192,
        max_tokens: int = 9600  # Maximum number of tokens per batch
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.token_vocab = token_vocab

        self.args = {'max_num_tokens':max_tokens,'num_buckets':6,'audio_max_length':20,'encodec_sr':50,'seed':1234}


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
            lengths_list=self.train_dataset.sizes, 
            verbose=True, 
            epoch=0)
        
        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=FlowDecoderCollator(audio_pad_token=self.token_vocab),
            num_workers=self.num_workers//world_size,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=FlowDecoderCollator_Val(audio_pad_token=self.token_vocab),
            num_workers=self.num_workers,
            persistent_workers=True,
        )