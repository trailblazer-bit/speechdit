import os
import sys
import glob
from tqdm import tqdm
import json
import random

save_dir = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/combine_mls_lrs3'
json1_fp = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/mls_en_witext/test_data.json'
json2_fp = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/data/lrs3/test_data.json'
all_data = []

with open(json1_fp,'r') as fp:
    data = json.load(fp)
    all_data.extend(data)

print(len(all_data))

with open(json2_fp,'r') as fp:
    data = json.load(fp)
    new_data = []
    for item in tqdm(data):
        item['code_len'] = item['v_frames']*2
        new_data.append(item)
    all_data.extend(new_data)

random.shuffle(all_data)
print(len(all_data))

with open(f'{save_dir}/test_data.json', 'w') as json_file:
    json.dump(all_data, json_file, indent=4)
