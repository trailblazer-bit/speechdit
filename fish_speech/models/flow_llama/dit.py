import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as L
from models.melae_kl_new import MelAEKL
from models.mel_vqgan import Mel_VQGAN
from models.speechdit import Diffusion, ModelArgs
from tasks.mel_vqgan import MelVQGAN_task
from utils.utils import ssim,weights_nonzero_speech
from Bigvgan.bigvgan.models_bigvgan import BigVGAN as BigVGANGenerator
from utils.ckpt import load_ckpt
from utils.plot import spec_to_figure
from utils.schedulers import WarmupSchedule
import numpy as np
from scipy.io import wavfile
from memory_profiler import profile
import gc



#Vocoder Config
config = {
    'model':{
        'resblock_kernel_sizes': [3, 7, 11],
        'resblock_dilation_sizes': [[1,3,5],[1,3,5],[1,3,5]],
        'upsample_rates': [5,2,2,2,2,2],
        'upsample_initial_channel': 1024,
        'upsample_kernel_sizes': [11,4,4,4,4,4]
    }
}


    
class DIT_task(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.h = hparams
        self.save_hyperparameters(hparams)
        #self.save_vae_latent = torch.empty(0, device=torch.device('cuda'))

        self.diff_model = self.build_model()
        
        self.ckpt_dir = 'Bigvgan/checkpoints/bigvgan_16k_librilight'
        self.vocoder = BigVGANGenerator(160, **config['model'])
        load_ckpt(self.vocoder, self.ckpt_dir, 'model_gen', silent=True)
        self.vocoder.remove_weight_norm()
        for param in self.vocoder.parameters():
            param.requires_grad = False
    
    def build_model(self):
        self.build_tts_model()
        self.gen_params = list(self.diff_model.parameters())
        #print(self.diff_model)
        #for n, m in self.diff_model.named_children():
        #    num_params(m, model_name=n)
        #if hparams['pitch_percep_ckpt'] != '':
        #    self.build_pitch_percep_model()
        return self.diff_model

   

    def build_tts_model(self):
        hparams = self.h
        #load_ckpt(self.vae_model, hparams['vae_path'], strict=True)

        if hparams.get('use_mel', False):
            self.vae_stride = 1
        if hparams.get('use_jzy_vae', False):
            #JZY latent
            checkpoint_path= '/home/xj_data/yangqian/TTS/lightning-framework/good-exp/jzy-vae/model_ckpt_steps_2210000.ckpt'
            self.vae_stride = 8
            self.vae_model = MelAEKL(hidden_size=256, vae_beta=1e-2, latent_dim=8, stride=8)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
            model_state_dict = checkpoint['state_dict']['model']
            self.vae_model.load_state_dict(model_state_dict)

        else:
            #VQ Latent
            checkpoint_path= '/home/xj_data/yangqian/TTS/lightning-framework/good-exp/0701-librilight-vae-gan-latent24-stride4/version_1/checkpoints/epoch=2-step=440000.ckpt'
            #self.vqvae_model = Mel_VQGAN(hidden_size=192, vae_beta=1e-2, latent_dim=8)
            self.vae_stride = hparams["vae_stride"]
            self.vae_model = MelAEKL(hidden_size=192, vae_beta=1e-2, latent_dim=24, stride=4)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
            model_state_dict = checkpoint['state_dict']
            #cleaned_state_dict = {k.replace('vqvae.', ''): v for k, v in model_state_dict.items() if k.startswith('vqvae.')}
            cleaned_state_dict = {k.replace('vae.', ''): v for k, v in model_state_dict.items() if k.startswith('vae.')}
            self.vae_model.load_state_dict(cleaned_state_dict)


        config = ModelArgs()
        config.target_type = 'epsilon' if hparams.get('use_ddpm', False) else 'velocity'
        config.target_type = 'vector_field' if hparams.get('use_vpcfm', False) else config.target_type
        config.use_expand_ph = hparams.get('use_expand_ph', False)
        config.zero_xt_prompt = hparams.get('zero_xt_prompt', False)
        self.diff_model = Diffusion(config, hparams)
        self.cfg_mask_token_phone = config.n_phone - 1
        self.cfg_mask_token_tone = config.n_tone - 1
        self.mask_r = hparams.get('mask_r', 0.5)


    def run_model(self, sample, infer=False, *args, **kwargs):
        hparams = self.h
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        #tone_tokens = sample["tone"]  # [B, T_t]
        txt_lengths = sample["txt_lengths"]  # [B, T_t]
        #bpe_tokens = sample["bpe_tokens"]  # [B, T_t]
        #bpe_lengths = sample["bpe_lengths"]  # [B, T_t]
        mels = sample["mels"]  # [B, T_s, 160]
        mel_lengths = sample["mel_lengths"]  # [B, T_s]
        #mel2ph = sample["mel2ph"][:, :: self.vae_stride]  # [B, T_mel/8]
        #latent_lengths = (mel2ph > 0).sum(-1)
        #loss_mask = (mel2ph > 0).float()[:, :, None]
        mels_loss_mask = (torch.abs(mels - (-6.0)) > 1e-3).float()
        mels_one_matrix = (mels * mels_loss_mask)[:, :: self.vae_stride].sum(-1)
        loss_mask = (torch.abs(mels_one_matrix - 0) > 1e-3).float()[:, :, None]
        #torch.set_printoptions(threshold=np.inf)

        #""" Disable the English tone (set them to 3)"""
        #en_tone_idx = ~((sample["tone"] == 4) | ( (11 <= sample["tone"]) & (sample["tone"] <= 15)) | (sample["tone"] == 0))
        #sample["tone"][en_tone_idx] = 3

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                if hparams.get('use_mel', False):
                    vae_latent = mels
                else:
                    vae_latent = self.vae_model.get_latent(mels).transpose(1,2)
                    #vae_latent = self.vqvae_model.get_latent(mels).transpose(1,2)
        
        #check_vae_latent = vae_latent.contiguous()
        #check_vae_latent = check_vae_latent.view(-1)
        #self.save_vae_latent = torch.cat((self.save_vae_latent, check_vae_latent))
        #if self.save_vae_latent.shape[0] > 100000:
        #    import matplotlib.pyplot as plt
        #    data = self.save_vae_latent.cpu().numpy()
        #    #plt.plot(data)
        #    plt.hist(data, bins=50, edgecolor='k', alpha=0.7)
        #    # 添加图表标题和标签
        #    plt.title('VAE(latent24-stride4) Latent Plot(num=1w)')
        #    plt.xlabel('Index')
        #    plt.ylabel('Value')
        #    plt.savefig('/mnt/workspace/yangqian/TTS/lightning-framework/figures/vae_value_stride4-latent24.png', format='png', dpi=300)
        #    breakpoint()

        if not infer:
            ''' Assign random CFG Mask in Training '''
            if hparams.get('use_seq_cfg', False):
                spk_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
                spk_cfg_mask = (spk_cfg_mask < 0.20).long()

                txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
                txt_cfg_mask = (txt_cfg_mask < 0.50).long() * spk_cfg_mask

                txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_token_phone * txt_cfg_mask
                #tone_tokens = tone_tokens * (1 - txt_cfg_mask) + self.cfg_mask_token_tone * txt_cfg_mask
                #bpe_tokens = bpe_tokens * (1 - txt_cfg_mask) + 32006 * txt_cfg_mask

            elif hparams.get('use_cfg', False):
                cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
                cfg_mask = (cfg_mask < 0.15).long() 
                txt_tokens = txt_tokens * (1 - cfg_mask) + self.cfg_mask_token_phone * cfg_mask
                #tone_tokens = tone_tokens * (1 - cfg_mask) + self.cfg_mask_token_tone * cfg_mask
                #bpe_tokens = bpe_tokens * (1 - cfg_mask) + 32006 * cfg_mask
            
            
            ctx_mask,latent_lengths = self.obtain_ctx_mask(mels, mels_loss_mask) # [B, T, 1]
            inputs = {
                'phone': txt_tokens,
                #'tone': tone_tokens,
                'text_lens': txt_lengths,
                #'bpe': bpe_tokens,
                #'bpe_lengths': bpe_lengths,
                'lat': vae_latent,
                'lat_lens': latent_lengths,
                'ctx_mask': ctx_mask,
                'lat_ctx': vae_latent * ctx_mask,
                'text_mel_mask': self.sequence_mask(latent_lengths + txt_lengths) > 0,  #txt_lengths+latent_lengths的 padding mask
                'spk_cfg_mask': spk_cfg_mask,

                #'cfg_mask': cfg_mask
                #'mel2ph': mel2ph
            }

            ''' Diffusion training '''
            ret_dict = self.diff_model(inputs)
            pred_v, target_v = ret_dict['pred_v'].transpose(1, 2), ret_dict['target_v'].transpose(1, 2)

            losses, output = {}, {}
            #算prompt部分的loss，只是mask了mel-padding loss
            losses["diff_loss"] = self.mse_loss(
                #pred_v * loss_mask * (1 - ctx_mask),
                pred_v * loss_mask,
                #target_v * loss_mask * (1 - ctx_mask),
                target_v * loss_mask,
            )

            if torch.any(torch.isnan(losses['diff_loss'])) or torch.any(torch.isinf(losses['diff_loss'])):
                print("NaN/INF occurs in loss")
                print(sample['item_name'])
                print('NaN! NaN! NaN! NaN! NaN! NaN! NaN! NaN!')
                exit(1)

            return losses, output
        else:
            ctx_mask,latent_lengths = self.obtain_ctx_mask(mels, mels_loss_mask, infer=True) # B, T, 1
            # Make CFG inputs
            txt_tokens_ = torch.cat([txt_tokens, txt_tokens, torch.full(txt_tokens.size(), self.cfg_mask_token_phone, device=txt_tokens.device)], 0)
            # tone_tokens_ = torch.cat([torch.full(tone_tokens.size(), self.cfg_mask_token, device=tone_tokens.device), tone_tokens], 0)
            # bpe_tokens_ = torch.cat([torch.full(bpe_tokens.size(), 32006, device=bpe_tokens.device), bpe_tokens], 0)

            vae_latent_ = vae_latent.repeat(3, 1, 1)
            txt_lengths_ = txt_lengths.repeat(3)
            # bpe_lengths_ = bpe_lengths.repeat(2)
            latent_lengths_ = latent_lengths.repeat(3)
            ctx_mask_ = ctx_mask.repeat(3,1,1)
            # mel2ph_ = mel2ph.repeat(2, 1)
            inputs = {
                'phone': txt_tokens_,
                #'tone': tone_tokens,
                'text_lens': txt_lengths_,
                #'bpe': bpe_tokens,
                #'bpe_lengths': bpe_lengths,
                'ctx_mask': ctx_mask_,
                'lat_ctx': vae_latent_ * ctx_mask_,
                'lat': vae_latent_,
                'lat_lens': latent_lengths_,
                #'text_mel_mask': self.sequence_mask(latent_lengths_ + txt_lengths_) > 0,
                #'mel2ph': mel2ph
                'dur': None,
            }

            x = self.diff_model.inference(inputs, timesteps=25, cfg_w=5, use_seq_cfg=True, seq_cfg_w=[3, 4.5])
            x = vae_latent * ctx_mask + x * (1 - ctx_mask)
            if hparams.get('use_mel', False):
                outputs = x
            else:
                outputs = self.vae_model.decode(x.transpose(1,2))
                #outputs = self.vqvae_model.decode(x.transpose(1,2))
            return outputs

    #@profile
    def training_step(self, batch, batch_idx):
        if not batch:
            return None
        loss_weights = {}
        loss_output, _ = self.run_model(batch)
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        #loss_output['batch_size'] = batch['txt_tokens'].size()[0]
        self.log('diff_loss', total_loss,prog_bar=True)
        return total_loss
    
    #@profile
    def validation_step(self, batch, batch_idx):
        if not batch:
            return None
        hparams = self.h
        loss_weights = {}
        model_out = self.run_model(batch, infer=True)
        #for loss_name, loss_value in loss_output.items():
        #    self.log(loss_name, loss_value,prog_bar=True)
        #total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        #loss_output['batch_size'] = batch['txt_tokens'].size()[0]
        
        with torch.no_grad():
            if hparams.get('use_mel', False):
                y_hat = model_out.detach().cpu()
            else:
                y_hat = mel_out = model_out.get('mel_out').detach().cpu()
            mels = batch["mels"]
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                    gt_vae_latent = self.vae_model.get_latent(mels).transpose(1,2)
                    #gt_vae_latent = self.vqvae_model.get_latent(mels).transpose(1,2)

            if hparams.get('use_mel', False):
                y = mels
            else:
                #y = self.vqvae_model.decode(gt_vae_latent.transpose(1,2)).get('mel_out').detach().cpu()
                y = self.vae_model.decode(gt_vae_latent.transpose(1,2)).get('mel_out').detach().cpu()

            if self.trainer.global_rank == 0 and batch_idx < 10:
                with torch.no_grad():
                    wav_gt = self.spec2wav(y.squeeze(0))
                    wav_gt = wav_gt.astype(float)
                    wav_gt = wav_gt * 32767
                    wav_recon = self.spec2wav(y_hat.squeeze(0))
                    wav_recon = wav_recon.astype(float)
                    wav_recon = wav_recon * 32767
                    #wav_gt = torch.from_numpy(wav_gt).unsqueeze(1)
                    self.plot_mel(batch_idx, y_hat, y)
                    import os
                    save_wav_dir = os.path.join(self.logger.log_dir, 'Val_Audio')
                    os.makedirs(save_wav_dir, exist_ok=True)
                    #print('saving... val...wav...')
                    wavfile.write(f'{save_wav_dir}/epoch_{self.current_epoch}_step_{self.global_step}_batch_{batch_idx}_gt.wav', 16000, wav_gt.astype(np.int16))
                    wavfile.write(f'{save_wav_dir}/epoch_{self.current_epoch}_step_{self.global_step}_batch_{batch_idx}_recon.wav', 16000, wav_recon.astype(np.int16))
                    #self.logger.experiment.add_audio(f'gt_audio_{batch_idx}', wav_gt, self.current_epoch, sample_rate=16000)
                    #self.logger.experiment.add_audio('pre_audio', wav_recon.unsqueeze(0), self.current_epoch, sample_rate=16000)
        
        #return total_loss
    
    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss
    
    
    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            if isinstance(mel, np.ndarray):
                mel = torch.FloatTensor(mel)
            mel = mel.unsqueeze(0).to(device)
            mel = mel.transpose(2, 1)
            y = self.vocoder.infer(mel).view(-1)
        y = y.to(torch.float32)
        wav_out = y.cpu().numpy()
        return wav_out

    def plot_mel(self, batch_idx, spec_out, spec_gt=None, name=None, title='', f0s=None, dur_info=None):
        #vmin = hparams['mel_vmin']
        #vmax = hparams['mel_vmax']
        if len(spec_out.shape) == 3:
            spec_out = spec_out[0]
        if isinstance(spec_out, torch.Tensor):
            spec_out = spec_out.to(torch.float32)
            spec_out = spec_out.cpu().numpy()
        if spec_gt is not None:
            if len(spec_gt.shape) == 3:
                spec_gt = spec_gt[0]
            if isinstance(spec_gt, torch.Tensor):
                spec_gt = spec_gt.to(torch.float32)
                spec_gt = spec_gt.cpu().numpy()
            max_len = max(len(spec_gt), len(spec_out))
            if max_len - len(spec_gt) > 0:
                spec_gt = np.pad(spec_gt, [[0, max_len - len(spec_gt)], [0, 0]], mode='constant',
                                 constant_values=vmin)
            if max_len - len(spec_out) > 0:
                spec_out = np.pad(spec_out, [[0, max_len - len(spec_out)], [0, 0]], mode='constant',
                                  constant_values=vmin)
            spec_out = np.concatenate([spec_out, spec_gt], -1)
        name = f'mel_val_{batch_idx}' if name is None else name
        self.logger.experiment.add_figure(name, spec_to_figure(
            spec_out,title=title), self.global_step)

    def obtain_ctx_mask(self, mels, loss_mask, infer=False, dtype=torch.float16):
        #mel2ph = sample["mel2ph"][:, :: self.vae_stride]  # [B, T_mel/8]
        #latent_lengths = (mel2ph > 0).sum(-1)          #[B, 去掉padding后的T_mel/8]; 目的是为了mask真实长度的50%
        #B, T = latent_lengths.shape[0], latent_lengths.max()   #T: txt_token_length
        mels = mels * loss_mask
        mels = mels[:, :: self.vae_stride].sum(-1)
        latent_lengths = (torch.abs(mels - 0) > 1e-3).sum(-1) 
        #B, T = latent_lengths.shape[0], latent_lengths.max()
        B, T = latent_lengths.shape[0], mels.shape[-1]
        device = mels.device

        if infer:
            # inference with 50% mask
            ratio = torch.ones((B, 1), device=device) * 0.5
        else:
            if self.h.get("random_ctx", True):
                ratio = torch.rand((B, 1), device=device) * self.mask_r
            else:
                ratio = torch.ones((B, 1), device=device) * self.mask_r

        end_idx = latent_lengths[:, None] * ratio
        end_idx = (end_idx - 1).clamp(min=1)
        ctx_mask = torch.arange(T, device=device)[None, :]
        ctx_mask = (ctx_mask < end_idx) # ctx_mask looks like: [1, 1, 1, 0, 0, 0, 0]
        ctx_mask = ctx_mask[:,:,None].to(dtype)
        #if not infer:
        #    breakpoint()
        return ctx_mask,latent_lengths

    def sequence_mask(self, seq_lens, max_len=None, device='cpu', dtype=torch.float16):
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(seq_lens.device) # [1, t]
        mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
        mask = mask.to(dtype)
        return mask
    
    def configure_optimizers(self):
        optimizer_gen = torch.optim.AdamW(
            self.gen_params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-07
        )
        return optimizer_gen
    
    def build_scheduler(self, optimizer):
        from utils.nn.schedulers import WarmupSchedule
        return WarmupSchedule(optimizer, 0.0001, 2000)

    def on_before_optimization(self, opt_idx):
        hparams = self.h
        if hparams.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.gradient_clip_norm)
        if hparams.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), hparams.gradient_clip_val)

    
