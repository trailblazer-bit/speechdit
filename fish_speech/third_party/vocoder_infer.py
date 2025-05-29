import glob
import os
import re
import traceback
import numpy as np
import torch
import torch.nn as nn
import librosa
import subprocess
from scipy.io import wavfile
from scipy.signal import get_window
import sys

sys.path.append('/apdcephfs_nj7/share_303172353/ggyzhang/projects/Bigvgan')
from bigvgan.models_bigvgan import BigVGAN as BigVGANGenerator


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path

def get_all_ckpts(work_dir, steps=None):
    if steps is None or steps == 0:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))

def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True, silent=False,
              load_opt=False, opts=None, steps=None):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if '.' not in model_name:
            state_dict = state_dict[model_name]
        else:
            base_model_name = model_name.split('.')[0]
            rest_model_name = model_name[len(base_model_name) + 1:]
            state_dict = {
                k[len(rest_model_name) + 1:]: v for k, v in state_dict[base_model_name].items()
                if k.startswith(f'{rest_model_name}.')}
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            cur_model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in cur_model_state_dict.items()}

            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        if not silent:
            print(f"| load '{model_name}' from '{ckpt_path}'.")
        if load_opt:
            optimizer_states = checkpoint['optimizer_states']
            assert len(opts) == len(optimizer_states)
            for optimizer, opt_state in zip(opts, optimizer_states):
                opt_state = {k.replace('_orig_mod.', ''): v for k, v in opt_state.items()}
                if optimizer is None:
                    return
                try:
                    optimizer.load_state_dict(opt_state)
                except ValueError:
                    print("| WARMING: optimizer parameters not match !!!")
        return checkpoint['global_step']
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)

def get_melspectrogram(wav_path, fft_size=800, hop_size=160, win_length=800, window="hann", num_mels=160,
    fmin=0, fmax=8000, eps=1e-6, sample_rate=16000, center=False, mel_basis=None):
    
    # Load wav
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    
    # Pad wav to the multiple of the win_length
    if len(wav) % win_length < win_length - 1:
        wav = np.pad(wav, (0, win_length - 1 - (len(wav) % win_length)), mode='constant', constant_values=0.0)
        
    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=center)
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else min(fmax, sample_rate // 2)
    if mel_basis is None:
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    if center:
        l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav = wav[:mel.shape[1] * hop_size]
    return {'wav': wav, 'mel': mel.T}

def save_wav(wav, path, sr):
    wav = wav.astype(float)
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])

ckpt_dir = '/workspace/user_code/kuachen/projects/Bigvgan/checkpoints/bigvgan_16k_librilight'
config = {
    'model':{
        'resblock_kernel_sizes': [3, 7, 11],
        'resblock_dilation_sizes': [[1,3,5],[1,3,5],[1,3,5]],
        'upsample_rates': [5,2,2,2,2,2],
        'upsample_initial_channel': 1024,
        'upsample_kernel_sizes': [11,4,4,4,4,4]
    }
}

class BigVGAN:
    def __init__(self, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = BigVGANGenerator(160, **config['model'])
        try:
            load_ckpt(self.model, ckpt_dir, 'model_gen', silent=True)
        except:
            traceback.print_exc()
        self.model.remove_weight_norm()
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            if isinstance(mel, np.ndarray):
                mel = torch.FloatTensor(mel)
            mel = mel.to(device) # (B,C,T)
            y = self.model.infer(mel).reshape(mel.shape[0],-1)
        # wav_out = y.cpu().numpy()
        return y

# vocoder = BigVGAN(device='cpu')
# acoustic_dict = get_melspectrogram('/workspace/user_code/kuachen/projects/fish-speech/prompt.wav')
# wav, mel = acoustic_dict['wav'], acoustic_dict['mel']
# mel = torch.from_numpy(mel)
# print(mel.shape)
# print(mel)
# mel = mel.unsqueeze(0).transpose(1,2)
# wav_recon = vocoder.spec2wav(mel)
# print(wav_recon.shape)
# save_wav(wav_recon, 'recon1.wav', sr=16000)
    