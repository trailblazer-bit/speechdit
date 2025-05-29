import torch
from torch import nn
from torch.nn import Linear

#from modules.asr.base import Prenet
#from modules.asr.seq2seq import TransformerASRDecoder
#from modules.commons.conformer.conformer import ConformerLayers
#from modules.commons.layers import LayerNorm, Embedding
#from modules.commons.transformer import SinusoidalPositionalEmbedding
#from utils.commons.hparams import hparams


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvBlock(nn.Module):
    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == 'ln':
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == 'wn':
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == 'none':
                pass
            elif self.norm == 'ln':
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn',
                 dropout=0, strides=None, res=True):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.res = res
        self.in_proj = Linear(idim, n_chans)
        if strides is None:
            strides = [1] * n_layers
        else:
            assert len(strides) == n_layers
        for idx in range(n_layers):
            self.conv.append(ConvBlock(
                n_chans, n_chans, kernel_size, stride=strides[idx], norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        hiddens = []
        for f in self.conv:
            x_ = f(x)
            x = x + x_ if self.res else x_  # (B, C, Tmax)
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)  # (B, Tmax, H)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)  # [B, L, C, T]
            return x, hiddens
        return x


class ConvGlobalStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0,
                 strides=[2, 2, 2, 2, 2]):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.pooling = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=strides[idx],
                                       norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x_nonpadding = (x.abs().sum(-1) > 0).float()
        x = self.in_proj(x) * x_nonpadding[..., None]
        x_nonpadding = x_nonpadding[:, None]
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x_nonpadding = x_nonpadding[:, :, ::2]
            x = f(x)  # (B, C, T)
            x = x * x_nonpadding
        x = x.transpose(1, -1)
        x = x.sum(1) / x_nonpadding.sum(-1)
        x = self.out_proj(x)  # (B, H)
        return x


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class VCASR(nn.Module):
    def __init__(self, dict_size, n_mel_bins=80):
        super().__init__()
        self.asr_enc_layers = hparams['asr_enc_layers']
        self.asr_dec_layers = hparams['asr_dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.num_heads = 2
        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=hparams['mel_strides'])
        if hparams['asr_enc_type'] == 'conv':
            self.content_encoder = ConvStacks(
                idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size)
        elif hparams['asr_enc_type'] == 'conformer':
            self.content_encoder = ConformerLayers(self.hidden_size, self.asr_enc_layers, 31,
                                                   use_last_norm=hparams['asr_last_norm'])
        self.token_embed = Embedding(dict_size, self.hidden_size, 0)
        self.asr_decoder = TransformerASRDecoder(
            self.hidden_size, self.asr_dec_layers, hparams['dropout'], dict_size,
            num_heads=self.num_heads)

    def forward(self, mel_input, prev_tokens=None):
        ret = {}
        h_content = ret['h_content'] = self.content_encoder(self.mel_prenet(mel_input)[1])
        if prev_tokens is not None:
            ret['tokens'], ret['asr_attn'] = self.asr_decoder(self.token_embed(prev_tokens), h_content)
        return ret
