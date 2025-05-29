# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

WORKERS_PER_NODE=64
WAV_DIR=/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3

python -m torch.distributed.run --nproc_per_node=$WORKERS_PER_NODE extract_speech_code.py \
    --wav_dir $WAV_DIR 
