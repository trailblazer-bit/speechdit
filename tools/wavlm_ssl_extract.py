import sys
import os
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0,'/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s')
from fish_speech.models.ssl_models.wavlm_extractor import Extractor
from fish_speech.models.ssl_models.utils.utils import read_wav_16k
from fish_speech.models.ssl_models.utils.espnet_utils import pad_list, make_pad_mask

wav_path = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/test-clean/61/70968/61-70968-0000.wav'
wav_path1 = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/test-clean/61/70970/61-70970-0001.wav'
model_path = '/apdcephfs_nj7/share_303172353/ggyzhang/projects/v2s/checkpoints/wavlm_large.pt'

codes = np.load('/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3_speech_token/audio/test/0Fi83BHQsMA/00002.npy')
print(codes.max(),codes.min())
exit(0)

extractor = Extractor(checkpoint=model_path, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
    output_layer=6)
    
audio = read_wav_16k(wav_path)
audio1 = read_wav_16k(wav_path1)
audio  = np.zeros((12800,))
audio1 = np.zeros((25600,))
print(audio.shape)
rep = extractor.extract_batch([audio,audio1],[len(audio),len(audio1)])
rep = rep.cpu().numpy()
print(rep.max(),rep.min())
print(rep.shape)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--wav-scp', type=str)
#     parser.add_argument("--out-dir", type=str)
#     parser.add_argument('--model', default="pretrained/WavLM-Large.pt", type=str)
#     parser.add_argument('--output-layer', default=6, type=int)
#     args = parser.parse_args()
    
#     extractor = Extractor(checkpoint=args.model, 
#                           device="cuda" if torch.cuda.is_available() else "cpu", 
#                           output_layer=args.output_layer)
    
#     out_dir=Path(args.out_dir).absolute()
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     with open(args.wav_scp, 'r') as f, torch.no_grad(), WriteHelper(f"ark,scp:{out_dir}/feats.ark,{out_dir}/feats.scp") as writer:
#         for line in tqdm(f.readlines()):
#             uttid, wav_path = line.strip().split(maxsplit=1)
#             logging.info("Extracting " + uttid)
#             audio = read_wav_16k(wav_path)
#             rep = extractor.extract(audio)
#             rep = rep.cpu().numpy()
#             writer(uttid, rep)