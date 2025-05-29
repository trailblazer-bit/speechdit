#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import torch
from tqdm import tqdm
import numpy as np

import torch.distributed as distr
import pathlib
from distributed import init_distributed_context

import logging
logger = logging.getLogger(__name__)
import os
import sys
import re
import glob

from huggingface_hub import snapshot_download

sys.path.insert(0,'/apdcephfs_nj7/share_303172353/ggyzhang/projects/Amphion')

from models.vc.vevo.vevo_utils import *



def single_job(infer_pipeline, wav_fp):
    tokens = inference_pipeline.extract_contentstyle_codes(wav_fp=wav_fp)
    return tokens.squeeze(0).numpy()


def extract_speech_token(args, rank, world_size):
    wavs = glob.glob(f'{args.wav_dir}/**/*.wav',recursive=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    # ===== Inference =====
    inference_pipeline = Vevo_ContentStyleTokenizer_Pipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        device=device,
    )

    print(len(wavs))
    for i in tqdm(range(rank, len(wavs), world_size)):
        wav_fp = wavs[i]
        item_name = os.path.basename(wav_fp).split('.')[0]
        new_fp = os.path.dirname(wav_fp).replace('LRS3','LRS3_speech_token')
        save_path = f'{new_fp}/{item_name}.npy'
        # if os.path.exists(save_path):
        #     continue
        try:
            speech_token = single_job(wav_fp)
        except:
            print('error!!!!!!!!',wav_fp)
            continue
        if len(speech_token)==0:
            continue
        os.makedirs(new_fp,exist_ok=True)
        np.save(f'{new_fp}/{item_name}.npy',speech_token)

def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        extract_speech_token(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str)
    parser.add_argument("--distributed_port", type=int, default=58564)
    args = parser.parse_args()

    main(args)