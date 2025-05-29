import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from fish_speech.models.dit_vc.mask import make_pad_mask
from torch.nn import CrossEntropyLoss
from torch.distributions import Categorical
from fish_speech.models.entropy_asr.decoding import PyTorchInference

torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)


class Vallina_ASR_Model(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 decoder_dim: int = 512,
                 audio_vocab_size: int = 500,
                 text_vocab_size: int = 32,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.decoder_dim = decoder_dim
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(audio_vocab_size, input_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), decoder_dim) if self.encoder.output_size() != decoder_dim else None


    def forward(self,codes,code_lengths,labels,label_lengths,texts,text_lengths):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)

        output = self.decoder(input_ids=texts,attention_mask=text_mask,encoder_outputs=h,\
            encoder_attention_mask=mask.squeeze(-1))

        loss = None
        logits = output[0]
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(logits.view(-1, self.text_vocab_size), labels.view(-1))
            loss_per_token = loss_per_token.view(batch_size, seq_length)
            masked_loss = loss_per_token * text_mask
            loss = masked_loss.sum() / text_mask.sum()

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,codes,code_lengths,text_bos_token=1):
        batch = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)

        bot_tokens = torch.full((batch,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference(h,mask.squeeze(-1),bot_tokens)
        return tokens
    
    @torch.inference_mode()
    def inference_topk_topp(self,codes,code_lengths,text_bos_token=1):
        batch = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)

        bot_tokens = torch.full((batch,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference_topk_topp(h,mask.squeeze(-1),bot_tokens)
        return tokens
    
    @torch.inference_mode()
    def inference_beam(self,codes,code_lengths,decoding_fn,rerank_fn,text_bos_token=1):
        batch = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)

        bot_tokens = torch.full((batch,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference_beam(h,mask.squeeze(-1),bot_tokens,decoding_fn,rerank_fn)
        return tokens
    
    @torch.inference_mode()
    def inference_tf(self,codes,code_lengths,labels,label_lengths,texts,text_lengths):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)

        output = self.decoder(input_ids=texts,attention_mask=text_mask,encoder_outputs=h,\
            encoder_attention_mask=mask.squeeze(-1))

        logits = output[0] # (B,T,vocab_size)

        probabilities = F.softmax(logits, dim=-1)
        next_tokens = Categorical(probabilities).sample() 
        return next_tokens

class Vallina_ASR_Whisper_Model(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 decoder_dim: int = 512,
                 audio_vocab_size: int = 500,
                 text_vocab_size: int = 32,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.decoder_dim = decoder_dim
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.encoder = encoder
        self.decoder = decoder

        self.input_embedding = nn.Embedding(audio_vocab_size, input_size)
        self.encoder_proj =  torch.nn.Linear(self.encoder.output_size(), decoder_dim) if self.encoder.output_size() != decoder_dim else None

        self.inference_ins = PyTorchInference(model=self.decoder)



    def forward(self,codes,code_lengths,labels,label_lengths,texts,text_lengths):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)

        logits = self.decoder(audio_features=h,tokens=texts)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(logits.view(-1, self.text_vocab_size), labels.view(-1))
            loss_per_token = loss_per_token.view(batch_size, seq_length)
            masked_loss = loss_per_token * text_mask
            loss = masked_loss.sum() / text_mask.sum()

        return {'loss': loss}
    
    @torch.inference_mode()
    def inference(self,codes,code_lengths,labels,label_lengths,texts,text_lengths):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)
 
        logits = self.decoder(audio_features=h,tokens=texts)
        print(logits)
        exit(0)

        probabilities = F.softmax(logits, dim=-1)
        next_tokens = Categorical(probabilities).sample() 
        return next_tokens
    
    @torch.inference_mode()
    def inference_topk_topp(self,codes,code_lengths,text_bos_token=1):
        batch = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
        token = self.input_embedding(torch.clamp(codes, min=0)) * mask

        h, h_lengths = self.encoder(token, code_lengths)
        if self.encoder_proj != None:
            h = self.encoder_proj(h)

        bot_tokens = torch.full((batch,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference_topk_topp(h,bot_tokens,self.inference_ins)
        return tokens


class Entropy_ASR_Model(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 decoder_dim: int = 512,
                 audio_vocab_size: int = 500,
                 text_vocab_size: int = 32,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 entropy_llm_ckpt: str = None,
                 entropy_llm_config = None,
                 patcher_args = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.decoder_dim = decoder_dim
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.encoder = encoder
        self.decoder = decoder
            

    def forward(self,codes,code_lengths,labels,label_lengths,texts,text_lengths,patch_ids,num_patches):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        # h = self.input_embedding(torch.clamp(codes, min=0)) * mask.unsqueeze(-1)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)

        # valid_patch_ids = patch_ids[:,code_lengths-1]
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        # valid_patch_ids_expanded = valid_patch_ids.unsqueeze(1)  # (B, 1)
        # h_mask = (torch.arange(h.size(1), device=h.device).expand(batch_size, -1) <= valid_patch_ids_expanded).float()


        output = self.decoder(input_ids=texts,attention_mask=text_mask,encoder_outputs=h,\
            encoder_attention_mask=h_mask)

        loss = None
        logits = output[0]
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(logits.view(-1, self.text_vocab_size), labels.view(-1))
            loss_per_token = loss_per_token.view(batch_size, seq_length)
            masked_loss = loss_per_token * text_mask
            loss = masked_loss.sum() / text_mask.sum()

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,codes,code_lengths,patch_ids,num_patches,text_bos_token=1):
        batch_size = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        # h = self.input_embedding(torch.clamp(codes, min=0)) * mask.unsqueeze(-1)

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )
        h = h_cross

        # valid_patch_ids = patch_ids[:,code_lengths-1]
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        # valid_patch_ids_expanded = valid_patch_ids.unsqueeze(1)  # (B, 1)
        # h_mask = (torch.arange(h.size(1), device=h.device).expand(batch_size, -1) <= valid_patch_ids_expanded).float()

        bot_tokens = torch.full((batch_size,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference(h,h_mask,bot_tokens)
        return tokens
    
    @torch.inference_mode()
    def inference_topk_topp(self,codes,code_lengths,patch_ids,num_patches,text_bos_token=1):
        batch_size = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        (h_encoder, h_cross), cache_encoder = self.encoder(
            tokens=codes,
            patch_embeds=None,
            cross_mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
        )

        h = h_cross
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        bot_tokens = torch.full((batch_size,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference_topk_topp(h,h_mask,bot_tokens)
        return tokens
    
    # @torch.inference_mode()
    # def inference_tf(self,codes,code_lengths,labels,label_lengths,texts,text_lengths):
    #     batch_size, seq_length = texts.shape[0], texts.shape[1]
    #     mask = (~make_pad_mask(code_lengths)).float().unsqueeze(-1).to(codes).float()
    #     token = self.input_embedding(torch.clamp(codes, min=0)) * mask

    #     h, h_lengths = self.encoder(token, code_lengths)
    #     if self.encoder_proj != None:
    #         h = self.encoder_proj(h)
        
    #     text_mask = (~make_pad_mask(text_lengths)).float().to(h)

    #     output = self.decoder(input_ids=texts,attention_mask=text_mask,encoder_outputs=h,\
    #         encoder_attention_mask=mask.squeeze(-1))

    #     logits = output[0] # (B,T,vocab_size)

    #     probabilities = F.softmax(logits, dim=-1)
    #     next_tokens = Categorical(probabilities).sample() 
    #     return next_tokens



class Entropy_ASR_Model_Merge(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 decoder_dim: int = 512,
                 audio_vocab_size: int = 500,
                 text_vocab_size: int = 32,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 entropy_llm_ckpt: str = None,
                 entropy_llm_config = None,
                 patcher_args = None,
                ):
        super().__init__()
        self.input_size = input_size
        self.decoder_dim = decoder_dim
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.encoder = encoder
        self.decoder = decoder
            

    def forward(self,codes,code_lengths,labels,label_lengths,texts,text_lengths,patch_ids,num_patches):
        batch_size, seq_length = texts.shape[0], texts.shape[1]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        patch_lens = patch_ids[torch.arange(patch_ids.size(0)), code_lengths - 1] + 1
        h = self.encoder(
            tokens=codes,
            patch_embeds=None,
            mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
            patch_lens=patch_lens,
        )
        # h_mask = (~make_pad_mask(patch_lens)).to(h).float()
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        
        text_mask = (~make_pad_mask(text_lengths)).float().to(h)

        # valid_patch_ids = patch_ids[:,code_lengths-1]
        # h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        # valid_patch_ids_expanded = valid_patch_ids.unsqueeze(1)  # (B, 1)
        # h_mask = (torch.arange(h.size(1), device=h.device).expand(batch_size, -1) <= valid_patch_ids_expanded).float()

        output = self.decoder(input_ids=texts,attention_mask=text_mask,encoder_outputs=h,\
            encoder_attention_mask=h_mask)

        loss = None
        logits = output[0]
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(logits.view(-1, self.text_vocab_size), labels.view(-1))
            loss_per_token = loss_per_token.view(batch_size, seq_length)
            masked_loss = loss_per_token * text_mask
            loss = masked_loss.sum() / text_mask.sum()

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,codes,code_lengths,patch_ids,num_patches,text_bos_token=1):
        batch_size = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        patch_lens = patch_ids[torch.arange(patch_ids.size(0)), code_lengths - 1] + 1
        h = self.encoder(
            tokens=codes,
            patch_embeds=None,
            mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
            patch_lens=patch_lens,
        )
        # h_mask = (~make_pad_mask(patch_lens)).to(h).float()
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)

        # valid_patch_ids = patch_ids[:,code_lengths-1]
        # h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)
        # valid_patch_ids_expanded = valid_patch_ids.unsqueeze(1)  # (B, 1)
        # h_mask = (torch.arange(h.size(1), device=h.device).expand(batch_size, -1) <= valid_patch_ids_expanded).float()

        bot_tokens = torch.full((batch_size,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference(h,h_mask,bot_tokens)
        return tokens
    
    @torch.inference_mode()
    def inference_topk_topp(self,codes,code_lengths,patch_ids,num_patches,text_bos_token=1):
        batch_size = codes.shape[0]
        mask = (~make_pad_mask(code_lengths)).to(codes).float()

        patch_lens = patch_ids[torch.arange(patch_ids.size(0)), code_lengths - 1] + 1
        h = self.encoder(
            tokens=codes,
            patch_embeds=None,
            mask=mask,
            num_patches=num_patches,
            patch_ids=patch_ids,
            patch_lens=patch_lens,
        )
        # h_mask = (~make_pad_mask(patch_lens)).to(h).float()
        h_mask = torch.ones((h.shape[0],h.shape[1])).to(h)

        bot_tokens = torch.full((batch_size,1),fill_value=text_bos_token).to(codes)
        tokens = self.decoder.inference_topk_topp(h,h_mask,bot_tokens)
        return tokens