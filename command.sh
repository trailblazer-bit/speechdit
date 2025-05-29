export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python fish_speech/train.py --config-name  v2s_unit_baseline.yaml

python fish_speech/train.py --config-name  v2s_tts_pretrain.yaml

python fish_speech/train.py --config-name  v2s_tts_pretrain_wostyle.yaml