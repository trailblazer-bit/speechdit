import torch
import sys

import os
from huggingface_hub import snapshot_download

sys.path.insert(0,'/apdcephfs_nj7/share_303172353/ggyzhang/projects/Amphion')

from models.vc.vevo.vevo_utils import *


def vevo_tts(
    src_text,
    ref_wav_path,
    timbre_ref_wav_path=None,
    output_path=None,
    ref_text=None,
    src_language="en",
    ref_language="en",
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== Device =====
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

    wav_path = "/apdcephfs_nj7/share_303172353/ggyzhang/projects/data/LRS3/audio/test/0Fi83BHQsMA/00002.wav"

    tokens = inference_pipeline.extract_contentstyle_codes(wav_fp=wav_path)
    print(tokens.shape)
