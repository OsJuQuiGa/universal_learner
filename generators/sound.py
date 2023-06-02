import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from generators.extra_models.utils import (ResConv1DBlock,
    ResConvTranspose1DBlock, PhaseShuffle2d, audio_range_amplitude, Null,
    PrintValues)

class Interpolate(nn.Module):
    def __init__(self, scale, mode = 'bilinear'):
        super().__init__()
        self.scale = scale
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x, 
            scale_factor = self.scale, 
            mode=self.mode)

class ResNet(nn.Module):
    def __init__(self, scale, ft, ft2, sampler, sampler_layer, conv_block):
        super().__init__()
        self.sampler = sampler
        if sampler == 'stride':
            self.inter_layer = Interpolate(scale)
        else:
            self.inter_layer = sampler_layer(None)
        self.conv_block = conv_block
        self.ft_adj = nn.Conv2d(ft, ft2, 1, 1)
    def forward(self, x):
        x_ch2 = self.ft_adj(x)
        return self.conv_block(x) + self.inter_layer(x_ch2)

class DownConvModel(nn.Module):
    def __init__(self, conv_kernels = ((25, 3),), stride_down = (4,2), res_net=False, 
            sampler='stride', max_ft = 128, end_ft = 32, target_flat_dim = 4096, 
            sampling = 2**15, channels=2, mimetic_filters = 32,
            norm_layer = None, phase_shuffle=4): #1x1 conv is done outside in mimetic form
        super().__init__()
        ''' dimesions of audio sample 
        '''
        downsampler = sampler
        ft_ch = end_ft
        ft_ch2 = end_ft
        flat_dim = target_flat_dim + 1
        min_ft = 16
        min_dim = 4
        downsamplings = torch.zeros(2, dtype=torch.long)
        conv_model = [
        #    nn.Conv2d(channels, ft_ch2, (conv_kernels[0][0], 7), 1),
        #    norm_layer(ft_ch2),
        #    nn.GELU()
        ]
        prev_flat = 0
        stride_down = list(stride_down)
        while(flat_dim > target_flat_dim):
            if downsamplings[0] == 0:
                ft_ch = channels
                ft_ch2 = end_ft
            else:
                ft_ch = max(min_ft,min(max_ft, ft_ch2))
                ft_ch2 = max(min_ft,min(max_ft, ft_ch*2))
            conv_layers = []
            if mimetic_filters <= min_dim:
                stride_down[1] = 1
            if sampling <= min_dim:
                stride_down[0] = 1
            scale = [1/stride_down[0], 1/stride_down[1]]
            for i,k in enumerate(conv_kernels):
                if downsampler == 'stride' and i == len(conv_kernels) - 1:
                    stride = stride_down
                else:
                    stride = [1, 1]
                if i == 0:
                    ft_conv_0 = ft_ch
                else:
                    ft_conv_0 = ft_ch2
                padding = (k[0]//2, k[1]//2)
                conv_layers += [
                    nn.Conv2d(ft_conv_0 , ft_ch2, k, stride=stride, padding=padding),
                    norm_layer(ft_ch2),
                    nn.GELU(),
                ]
                if phase_shuffle > 0:
                    conv_layers+= [PhaseShuffle2d(phase_shuffle)]
            if downsampler == 'maxpool':
                down_layer_fn = lambda x: nn.MaxPool2d(5,stride_down, padding=5//2)
            elif downsampler == 'avgpool':
                down_layer_fn = lambda x: nn.AvgPool2d(5,stride_down, padding=5//2)
            elif downsampler in ['bilinear', 'area', 'nearest']:
                down_layer_fn = lambda x: Interpolate(scale, downsampler)
            elif downsampler == 'stride':
                down_layer_fn = lambda x: Null()
            conv_layers.append(down_layer_fn(None))
            conv_block = nn.Sequential(*conv_layers)
            if res_net:
                conv_block = ResNet(
                    scale, ft_ch, ft_ch2, downsampler, down_layer_fn, conv_block
                )
            conv_model.append(conv_block)
            sampling = sampling//stride_down[0]
            mimetic_filters = mimetic_filters//stride_down[1]
            flat_dim = ft_ch2 * sampling * mimetic_filters
            downsamplings += torch.tensor(
                [stride_down[0]>1, stride_down[1]>1]
            )
            if prev_flat == flat_dim:
                break    
            prev_flat = flat_dim
        self.model = nn.Sequential(*conv_model)
        self.conv_out_dim = (ft_ch2, sampling, mimetic_filters)
        self.flat_dim = flat_dim
        self.downsamplings = downsamplings
        
    def forward(self, x):
        return self.model(x)

class UpConvModel(nn.Module):
    def __init__(self, conv_kernels = [(25, 3)], stride_up = (4,2), res_net=True, 
            sampler='stride', max_ft = 128, end_ft = 32, target_flat_dim = 4096, 
            sampling = 2**15, channels=2, mimetic_filters = 32,
            norm_layer = None): 
        super().__init__()
        upsampler = sampler
        down_model = DownConvModel(conv_kernels, stride_down =stride_up, res_net=res_net, 
            sampler='bilinear', max_ft = max_ft, end_ft = end_ft, target_flat_dim = target_flat_dim, 
            sampling = sampling, channels=channels, mimetic_filters = mimetic_filters,
            norm_layer = norm_layer, phase_shuffle=0)
        self.flat_dim = down_model.flat_dim
        self.init_conv_dim = down_model.conv_out_dim
        upsamplings = down_model.downsamplings
        maxdimup = torch.max(upsamplings).item()
        del(down_model)
        sampling = self.init_conv_dim[1]
        mimetic_filters = self.init_conv_dim[2]

        min_ft = 16
        #end_ft = init_conv_dim[0]
        conv_model = []
        for i in range(maxdimup):
            ft = int(2**(np.log2(end_ft) + maxdimup - i))
            ft_ch = max(min_ft,min(max_ft, ft))
            ft_ch2 = max(min_ft,min(max_ft, ft//2))
            stride = [
                stride_up[0] if upsamplings[0] - maxdimup + i > -1 else 1,
                stride_up[1] if upsamplings[1] - maxdimup + i > -1 else 1
            ]
            conv_layers = []
            #if i > 0:
            if upsampler == 'maxpool': #maxpool
                up_layer_fn = lambda x : Interpolate(stride, 'nearest')
            elif upsampler in ['bilinear', 'nearest', 'area']:
                up_layer_fn = lambda x : Interpolate(stride, upsampler)
            elif upsampler == 'stride':
                up_layer_fn = lambda x : Null()
            conv_layers.append(up_layer_fn(None))
            for j,k in enumerate(conv_kernels):
                if j==0:
                    ft_conv_0 = ft_ch
                else:
                    ft_conv_0 = ft_ch2
                padding = (k[0]//2, k[1]//2)
                if upsampler == 'stride' and j == 0:
                    conv_mod = nn.ConvTranspose2d(
                        ft_conv_0 , ft_ch2, k, stride=stride, 
                        padding=padding, output_padding=3
                    )
                else:
                    conv_mod = nn.Conv2d(
                        ft_conv_0 , ft_ch2, k, stride=(1, 1), 
                        padding=padding,
                    )
                conv_layers += [
                    conv_mod,
                    norm_layer(ft_ch2),
                    nn.GELU(),
                ]
            conv_block = nn.Sequential(*conv_layers)
            if res_net: #and i>0:
                conv_block = ResNet(2, ft_ch, 
                    ft_ch2, upsampler, up_layer_fn, conv_block)
            conv_model.append(conv_block)
            sampling = sampling*stride[0]
            mimetic_filters = mimetic_filters*stride[1]
        conv_model += [
            nn.Conv2d(
                ft_ch2,channels, 
                kernel_size = (25, mimetic_filters),
                stride = (1, mimetic_filters//2), 
                padding = (25//2, 0)), # B, 2, fs, mft
            nn.Flatten(-2,-1),
            #norm_layer(ft_ch2),
            nn.Tanh()
        ]
        assert end_ft == ft_ch2
        self.model = nn.Sequential(*conv_model)
    def forward(self,x):
        return self.model(x)

class Sound2Latent(nn.Module):
    '''Converts sound to latent space, the order of downsamplings
        and residual blocks are user defined, then flattens
    '''
    def __init__(self,
        batch_size = 1, max_ft = 256, target_flat_dim = 256, latent_dim = 512,
        kernel_size = (25, 3), channels = 2, sampling = 2**15, note_div = 12,
        mimetic_filters = 64, end_ft = 32, mimetic_kernel = 33, device='cpu',seq_len = 20):
        '''model_order:list[str]: 'io' input downsampling,
            'down' downsampling, 'res' res block
            strides: list[int]: strides of downsamplings
            batch_size: int
            feature_channels: int: number of convolutional filters
            in_dim: int:
            our_dim: int:
            inter_dims: list:
            channels: int: input channels
            d: int: duration of recording
            fs: int: samples per second
        '''
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sampling = sampling
        self.mimetic = mimetic_kernel
        self.device = device
        self.init_conv_filters = self.get_init_filters(
            channels, mimetic_filters, note_div, mimetic_kernel, sampling 
        )
        conv_model = [DownConvModel(
            conv_kernels=[kernel_size], stride_down=(4,2), sampler='bilinear',
            max_ft=max_ft, end_ft=end_ft, target_flat_dim=target_flat_dim, 
            sampling=sampling, channels = channels,
            norm_layer=nn.InstanceNorm2d, phase_shuffle=2
        )]
        fdim = batch_size*conv_model[-1].flat_dim
        hdim = (fdim+latent_dim)//2
        conv_model+=[
            nn.Flatten(0, -1),
            nn.Unflatten(0, (1, fdim)),
            nn.Linear(fdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, latent_dim),
        ]
        self.conv_model = nn.Sequential(*conv_model)

    def get_init_filters(self, channels, ft_ch, note_div = 12, kernel_size = 33,
            sampling=2**15, freq_range =(20,20000+1024)):
        ''' kernel_size = max kernel size
        '''
        max_freq = min(sampling//2, freq_range[1])
        base_freq = 27.5/2 #for 440hz tuning
         #(kernel_size, dilation, convfilter) with padding function
        i = 0
        frequencies = []
        freq = base_freq
        while(freq < max_freq):
            freq = base_freq*2**(i/note_div)
            if freq > freq_range[0]:
                frequencies.append(freq)
            i += 1
        assert len(frequencies) >= ft_ch
        frequencies = frequencies[-ft_ch:]
        conv_filters = []
        conv_param = []
        for freq in frequencies:
            t = 1/freq
            samples = t * sampling * 4
            kernels_in_sample = samples/kernel_size
            if kernels_in_sample <= (1.0+1e-6):
                k = math.ceil(samples)//2*2 + 1
                dilation = 1
            else:
                k = kernel_size
                dilation = int(kernels_in_sample)
            samples_total = 1 + dilation*(k - 1)
            padding = (samples_total//2)

            waves_cont = samples / samples_total
            with torch.no_grad():
                x = torch.linspace( #div by 2 for each size, but 2*pi for wave
                    -waves_cont, waves_cont, k).to(device=self.device)
                xa = torch.linspace(-1, 1, k).to(device=self.device)
                y = torch.cos(x) * (1 - torch.abs(torch.cos(xa))) #* audio_range_amplitude(freq)
                filter_wgt = torch.zeros(
                    [channels, channels, y.size()[0]], device=self.device)
                for ch in range(channels):
                    filter_wgt[ch, ch] += y.clone()
            conv_filters.append(filter_wgt)
            conv_param.append((k, dilation, padding, freq))
        
        return list(zip(conv_filters, conv_param))

    def forward(self, x: torch.Tensor, **kwargs:dict): #B, C, S
        x_init = []
        for cfilter, cparam in self.init_conv_filters:
            _, dilation, padding, _ = cparam
            padded = F.pad(x, pad=(padding,padding), mode='constant')
            convded = F.conv1d(padded, cfilter, stride=1,
                dilation=dilation)
            x_init.append(convded)
        x_init = torch.stack(x_init, dim=3) #B, ch, samples, freq
        x_flat = self.conv_model(x_init) #1, lat
        return x_flat

class Latent2Sound(nn.Module):
    '''Coverts latent space to sound. The order of upsamplings
        and residual blocks are user defined
    '''
    def __init__(self,
        batch_size = 1, max_ft = 256,  target_flat_dim = 256, latent_dim = 512, kernel_size = 61,
        channels = 2, sampling = 44100, end_ft = 32, mimetic_filters = 64, seq_len = 20,):
        #strides are in the reverse order
        '''model_order:list[str]: 'io' input downsampling,
            'down' downsampling, 'res' res block
            strides: list[int]: strides of upsamplings, reverse order because
                is using downsampling order
            batch_size: int
            feature_channels: int: number of convolutional filters
            in_dim: int:
            our_dim: int:
            inter_dims: list:
            channels: int: input channels
            d: int: duration of recording
            fs: int: samples per second
        '''
        super().__init__()
        #print([batch_size, in_dim, out_dim])
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sampling = sampling
        self.channels = channels
        up_conv = UpConvModel(conv_kernels=[kernel_size], stride_up=(4,2), res_net=True,
            sampler='bilinear', norm_layer=nn.InstanceNorm2d, max_ft=max_ft, end_ft=end_ft,
            target_flat_dim = target_flat_dim, mimetic_filters= mimetic_filters,
            sampling = sampling, channels=channels)
        fdim = batch_size*up_conv.flat_dim
        hdim = (fdim+latent_dim)//2
        conv_model = [
            nn.Linear(latent_dim, hdim),
            nn.GELU(),
            nn.Linear(hdim, fdim),    
            nn.GELU(),
            nn.Unflatten(1, (batch_size, *up_conv.init_conv_dim)),
            nn.Flatten(0, 1),
            up_conv
        ]
        self.conv_model = nn.Sequential(*conv_model)
    def forward(self, x, **kwargs):
        #B, latcell
        return self.conv_model(x), {} #B, C, S



