import os
from tqdm import tqdm
import glob
import numpy as np
import random
import json
import torch

import numpy as np


audio_dir = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3/audio'
mouth_dir = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3_preprocess/lrs3/lrs3_video_seg16s'
token_dir = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3_speech_token/audio'
audio_map = {}
mouth_map = {}
token_map = {}
data = []
 
wavs = glob.glob(f'{audio_dir}/**/*.wav',recursive=True)
print(len(wavs))
for wav_fp in tqdm(wavs):
    tmp_wav_fp = wav_fp.split('.')[0]
    fid = '/'.join(tmp_wav_fp.split('/')[-3:])
    audio_map[fid] = wav_fp


tokens = glob.glob(f'{token_dir}/**/*.npy',recursive=True)
print(len(tokens))
for token_fp in tqdm(tokens):
    tmp_token_fp = token_fp.split('.')[0]
    fid = '/'.join(tmp_token_fp.split('/')[-3:])
    token_map[fid] = token_fp

mouths = glob.glob(f'{mouth_dir}/**/*.mp4',recursive=True)
print(len(mouths))
for mouth_fp in tqdm(mouths):
    tmp_mouth_fp = mouth_fp.split('.')[0]
    fid = '/'.join(tmp_mouth_fp.split('/')[-3:])
    audio_fn = audio_map.get(fid,None)
    unit_fn = token_map.get(fid,None)
    if audio_fn is None or unit_fn is None:
        continue
    v_frames = len(np.load(unit_fn))//2
    item = {'fid':fid,'wav_fn':audio_fn,'video_fn':mouth_fp,'unit_fn':unit_fn,'v_frames':v_frames}
    data.append(item)

print(len(data))

random.shuffle(data)
print(len(data))
with open('/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/lrs3/train_data.json', 'w') as json_file:
    json.dump(data[:-80], json_file, indent=4)

with open('/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/lrs3/valid_data.json', 'w') as json_file:
    json.dump(data[-80:-40], json_file, indent=4)

with open('/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/lrs3/test_data.json', 'w') as json_file:
    json.dump(data[-40:], json_file, indent=4)

exit(0)


