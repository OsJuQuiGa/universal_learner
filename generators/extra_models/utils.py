#from ast import Pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import (get_io_slices, 
    param_function, param_function_return)

class Print(nn.Module):
    ''' Dummy torch.nn module that prints intermediate values of the main model
        in console
    '''
    def forward(self, x):
        print(x)
        return x
class PrintSize(nn.Module):
    ''' Dummy torch.nn module that prints intermediate values of the main model
        in console
    '''
    def forward(self, x):
        print(x.size())
        return x
class PrintValues(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self, x):
        if len(self.args)>0:
            print(self.args)
        if len(self.kwargs)>0:
            print(self.kwargs)
        return x

class Null(nn.Module):
    def forward(self, x):
        return x

class CellModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 0
        self.mtype = None
    def get_out_dim(self,):
        return self.out_dim
    def next_cycle(self):
        pass

class CellModelWithParameters(CellModelBase):
    def __init__(self, dim_param, param_names, lat_param) -> None:
        super().__init__()
        self.inlat = sum(lat_param.values())
        self.indim = sum(dim_param.values())
        self.param_names = param_names
        self.slices = get_io_slices([dim_param, lat_param], param_names)

    def in2lat(self, parameters, pre_size:tuple, device:torch.device):
        in_val = torch.zeros(
            (*pre_size, self.indim), device=device
        )
        pslen = len(pre_size)
        def tmp(pn):
            torch.index_select(in_val)
            if pslen == 1:
                to_add = in_val[:, self.slices[pn][0]]
                in_val[:, self.slices[pn][0]] = to_add + parameters[pn]
            elif pslen == 2:
                to_add = in_val[:, :, self.slices[pn][0]]
                in_val[:, :, self.slices[pn][0]] = to_add + parameters[pn]
        param_function(self, p_func=tmp)
        return in_val

    def lat2in(self, latent, pre_size:tuple()):
        pslen = len(pre_size)
        def tmp(pn):
            if pslen == 1:
                sliced = latent[:, self.slices[pn][1]]
            elif pslen == 2:
                sliced = latent[:, :, self.slices[pn][1]]
            return sliced
        return param_function_return(self, p_func=tmp)

    def in2term(self, parameters):
        pk0 = self.param_names[0]
        values = parameters[pk0]
        del(parameters[pk0])
        return values, parameters

def audio_range_amplitude(freq):
    amp_range = (
        (19.99, 0.112),
        (100, 0.635),
        (200, 0.844),
        (1000, 0.943),
        (1500, 0.911),
        (3500, 1.0),
        (12000, 0.748),
        (18000, 0.781),
        (21000, 0.000)
    )
    y = 0.0
    for i, dup in enumerate(amp_range):
        if freq > dup[0]:
            continue
        else:
            dup0 = amp_range[i-1]
            s = (dup[1] - dup0[1])/(dup[0]-dup0[0])
            y = dup0[1] + s*(freq - dup0[0])
            break
    return y

def set_learn_functions(lr = 0.01, 
    optimizer_type = 'Adam', lf_type = 'MSE', 
    model=None, decay = 0.01):
    '''sets the optimizer and the loss function
        lr: float: learning rate
        optimizer_type: str: type of optimizer
        lf_type: str: type of loss function
        model: torch.nn.module: torch model to bind 
            its parameters to the optimizer
        decay: float: weight decay normalization, keeps the magnitude of weights and
            biases low otherwise
        ---- return (
            optimizer: torch.nn.optim
            lossfunction: torch.F
        )
    '''
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=decay) 
    if optimizer_type == 'SGDmom':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay=decay) 
    if optimizer_type == 'AdaGrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr = lr, weight_decay=decay)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=decay)
    if optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = lr, weight_decay=decay)
    
    if lf_type == 'MSE':
        lossfunction = torch.nn.MSELoss()
    if lf_type == 'SL1':
        lossfunction = torch.nn.SmoothL1Loss()
    if lf_type == 'L1':
        lossfunction = torch.nn.L1Loss()

    return optimizer, lossfunction
def TanhX(x):
    '''it reduces the magnitude of input before tanh operation and later 
        restores the magnitude by the same scaling value, it makes tanh less prone
        to saturation
        x:float: scaling value
        ---- return function
    '''
    return lambda v : x*torch.tanh(v/x)

def SigmoidX(x):
    ''' the same as TanhX but for sigmoid
    '''
    return lambda v : x*torch.sigmoid(v/x)

def str2act(s):
    '''converts string to non-linear activation function
        s:str
        ---- return torch.F: activation function
    '''
    if s == 'none':
        return lambda x: x
    elif s == 'hardtanh':
        return nn.Hardtanh()
    elif s == 'sigmoid':
        return nn.Sigmoid()
    elif s == 'relu6':
        return nn.ReLU6()
    elif s == 'tanh':
        return nn.Tanh()
    elif s == 'tanhshrink':
        return nn.Tanhshrink()
    elif s == 'hardshrink':
        return nn.Hardshrink()
    elif s == 'GELU':
        return nn.GELU()
    elif s == 'softshrink':
        return nn.Softshrink()
    elif s == 'softsign':
        return nn.Softsign()
    elif s == 'relu':
        return nn.ReLU()
    elif s == 'prelu':
        return nn.PReLU()
    elif s == 'softplus':
        return nn.Softplus()
    elif s == 'elu':
        return nn.ELU()
    elif s == 'selu':
        return nn.SELU()
    else:
        raise ValueError("[!] Invalid activation function.")

class ResnetBlock(nn.Module):
    #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, res_scale=1.0):
        '''Residual block for one dimensional tensors
            n_in: int: input channels
            n_state: int: intermediate channels inside the block
            dilation: int: dilation of the kernel
            res_scale: float: scale of the residue
        '''
        #https://github.com/openai/jukebox/blob/master/jukebox/vqvae/resnet.py
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.Conv1d(n_in, n_state, 3, 1, 
                padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
            nn.ReLU(),
        )
        self.res_scale = res_scale

    def forward(self, x):
        '''---- return (B, L, n_in)
        '''
        return x + self.res_scale * self.model(x)


class ResConvTranspose1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, res_scale=1.0):
        '''The same as ResConv1DBlock but with convolutional transpose
        '''
        #https://github.com/openai/jukebox/blob/master/jukebox/vqvae/resnet.py
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ConvTranspose1d(n_state, n_in, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(n_in, n_state, 3, 1, 
                padding, dilation = dilation),
            nn.ReLU(),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)        

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class PhaseShuffle2d(nn.Module):
    def __init__(self, rad=2, pad_type='reflect'):
        super().__init__()
        self.rad = rad
        self.pad_type = pad_type

    def forward(self, x):
        b, ch, x_len, y_ft = x.size()

        phase = torch.randint(low=-self.rad, high=self.rad+1, size = (1,))
        pad_l = max(phase, 0)
        pad_r = max(-phase, 0)
        phase_start = pad_r
        padded = F.pad(x, (0, 0, pad_l, pad_r), mode=self.pad_type)

        shifted = padded[:, :, phase_start:phase_start+x_len]
        #x.view(b, x_len, ch)
        return shifted


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())

def init_weights(model):
    assert isinstance(model, nn.Module)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

