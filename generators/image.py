import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from generators.extra_models.utils import (CellModelBase, ResnetBlock, 
    PrintValues, PrintSize, Null)
from utils import transform2affine_pyt


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
    ''' Latent space config:
        8192: 512, 4x4
        4096: 1024, 2x2
        2048: 512, 2x2
        1024: 256, 2x2
        512: 128, 2x2
    '''
    def __init__(self, conv_kernels = [3,3], res_blocks=0, res_blocks_end=0, 
            res_net=False, sampler='stride', max_ft = 512, init_ft = 64, 
            target_flat_dim = 4096, img_size = 64, channels=3, 
            norm_layer = None): #1x1 conv is done outside
        super().__init__()
        downsampler = sampler
        ft_ch = init_ft
        ft_ch2 = init_ft
        res = img_size
        flat_dim = target_flat_dim + 1
        conv_model = [
            nn.Conv2d(channels, ft_ch, 1, 1),
            norm_layer(ft_ch),
            nn.GELU()]
        downsamplings = 0
        min_ft = 4
        min_res = 2
        prev_flat = 0
        while(flat_dim > target_flat_dim):
            ft_ch = max(min_ft,min(max_ft, ft_ch2))
            ft_ch2 = max(min_ft,min(max_ft, ft_ch*2))
            conv_layers = []
            for i,k in enumerate(conv_kernels):
                if downsampler == 'stride' and i == len(conv_kernels) - 1:
                    stride = 2
                else:
                    stride = 1
                if i == 0:
                    ft_conv_0 = ft_ch
                else:
                    ft_conv_0 = ft_ch2
                
                conv_layers += [
                    nn.Conv2d(ft_conv_0 , ft_ch2, k, stride, padding=k//2),
                    norm_layer(ft_ch2),
                    nn.GELU(),
                ]
            for _ in range(res_blocks):
                conv_layers.append(ResnetBlock(
                    ft_ch2, 'reflect', norm_layer, False, True))

            if downsampler == 'maxpool':
                down_layer_fn = lambda x: nn.MaxPool2d(2,2)
            elif downsampler in ['bilinear', 'area', 'nearest']:
                down_layer_fn = lambda x: Interpolate(1/2, downsampler)
            elif downsampler == 'stride':
                down_layer_fn = lambda x: Null()
            conv_layers.append(down_layer_fn(None))
            conv_block = nn.Sequential(*conv_layers)
            if res_net:
                conv_block = ResNet(
                    1/2, ft_ch, ft_ch2, downsampler, down_layer_fn, conv_block
                )
            conv_model.append(conv_block)
            res = res//2
            flat_dim = ft_ch2 * res**2
            downsamplings += 1
            if res < min_res:
                break
        #if flat_dim != target_flat_dim:
            
        for _ in range(res_blocks_end):
            conv_model.append(ResnetBlock(
                ft_ch2, 'reflect', norm_layer, False, True))
        self.model = nn.Sequential(*conv_model)
        self.conv_out_dim = (ft_ch2, res, res)
        self.flat_dim = flat_dim
        self.downsamplings = downsamplings
        if self.flat_dim != target_flat_dim:
            print('Flat dim '+str(self.flat_dim)+' different from target '\
                +str(target_flat_dim))

    def forward(self, x):
        return self.model(x)

            
class UpConvModel(nn.Module):
    def __init__(self, conv_kernels = [3,3], res_blocks=0, res_blocks_start=0, res_net=False, 
            sampler='stride',max_ft = 512, end_ft = 64, min_ft =16, target_flat_dim = 4096, 
            img_size = 64, channels=3, norm_layer = None,):
        super().__init__()
        upsampler = sampler
        down_model = DownConvModel(conv_kernels, 0, 0, False, sampler='bilinear', 
            max_ft = max_ft, init_ft=end_ft, target_flat_dim=target_flat_dim,
            img_size = img_size, norm_layer=norm_layer)
        self.flat_dim = down_model.flat_dim
        self.init_conv_dim = down_model.conv_out_dim
        upsamplings = down_model.downsamplings
        del(down_model)
        res = self.init_conv_dim[1]
        #init_ft = init_conv_dim[0]
        conv_model = []
        for i in range(upsamplings):
            ft = int(2**(np.log2(end_ft) + upsamplings - i))
            ft_ch = max(min_ft,min(max_ft, ft))
            ft_ch2 = max(min_ft,min(max_ft, ft//2))
            conv_layers = []
            #if i > 0:
            if upsampler == 'maxpool': #maxpool
                up_layer_fn = lambda x : Interpolate(2, 'nearest')
            elif upsampler in ['bilinear', 'nearest', 'area']:
                up_layer_fn = lambda x : Interpolate(2, upsampler)
            elif upsampler == 'stride':
                up_layer_fn = lambda x : Null()
            conv_layers.append(up_layer_fn(None))
            if i == 0:
                for _ in range(res_blocks_start): 
                    conv_model.append(ResnetBlock(
                        ft_ch, 'reflect', norm_layer, False, True))
            for _ in range(res_blocks):
                conv_model.append(ResnetBlock(
                        ft_ch, 'reflect', norm_layer, False,True))
            for j,k in enumerate(conv_kernels):
                #if j == len(conv_kernels) - 1:
                if j==0:
                    ft_conv_0 = ft_ch
                else:
                    ft_conv_0 = ft_ch2
                if upsampler == 'stride' and j == 0:
                    conv_mod = nn.ConvTranspose2d(
                        ft_conv_0 , ft_ch2, k, 2, padding=k//2, output_padding=3
                    )
                else:
                    conv_mod = nn.Conv2d(
                        ft_conv_0 , ft_ch2, k, 1, padding=k//2,
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
            res = res*2
        conv_model += [
            nn.Conv2d(ft_ch2,channels, 7, padding=7//2),
            #norm_layer(ft_ch2),
            nn.Tanh()
        ]
        assert end_ft == ft_ch2
        self.model = nn.Sequential(*conv_model)
    def forward(self,x):
        return self.model(x)

class STRetina(CellModelBase):
    def __init__(self, use_affine_transform, size, channels, norm_layer, ex_lat):
        super().__init__()     
        if use_affine_transform:
            self.localization  = DownConvModel(conv_kernels=[3], res_blocks=0, res_blocks_end=0,
                res_net=False, sampler='maxpool', max_ft=16, init_ft=4,
                target_flat_dim=256, img_size=size, channels= channels,
                norm_layer=norm_layer)
            self.ex_lat = ex_lat
            fc_loc = [
                nn.Linear(self.localization.flat_dim + ex_lat, 64),
                nn.GELU(),
                *[nn.Linear(64,64),nn.GELU()]*2,
                nn.Linear(64, 7)
            ]
            fc_loc[-1].weight.data.zero_()
            fc_loc[-1].bias.data.copy_(
                torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                #rotation(1), scale(2), location(2), shear(2)
            )
            self.fc_loc = nn.Sequential(*fc_loc)
            self.st_grid = self.stn
            self.st_t = lambda x, grid: F.grid_sample(x, 
                grid, padding_mode='reflection', align_corners=False)
        else:
            self.st_grid = lambda x, y : None
            self.st_t = lambda x, y : x
            
    def stn(self, x, ex_lat):
        xs = self.localization(x)
        xs = xs.view(-1, self.localization.flat_dim)#B, lat
        if self.ex_lat>0:
            xs = torch.cat([xs, ex_lat], dim=1)
        psi = self.fc_loc(xs)
        theta = transform2affine_pyt(
            psi[:,0], psi[:,[1,2]], psi[:,[3,4]], psi[:,[5,6]],
            limits='intrinsic'
        )
        grid = F.affine_grid(theta, x.size())
        #x = F.grid_sample(x, grid)
        return grid

class Retina2Latent(STRetina):
    def __init__(self, size = 64, views = [1,4,16], batch_size = 1,
        ft_ch = [64,32,16], latent_dim = 1024, sampler = [],
        target_flat_dim = [4096, 2048, 1024],
        channels=3, latent_layers = 3, ex_transform_dim = 5,
        use_affine_transform = False):
        ''' Model class for converting images from nviews to latent space
            it makes downsamplings then followed up by residual blocks, then
            flattening
            views: list: scales of the image
            size: int: size of the kernel filter
            batch_size: int:
            feature_channels: int: gets duplicated for each downsampling
            in_dim: list: input dimensions
            out_dim: list: output dimensions
            inter_dims: list: inter dimensions
            channels: int: RGB channels
            downsamplings: int:
            res_blocks: int:
        '''
        
        #print([batch_size,*in_dim, out_dim])
        
        self.batch_size = batch_size
        self.views = views
        self.len_view = len(views)
        self.size = size
        self.out_dim = latent_dim
        ex_lat = 32
        norm_layer = nn.InstanceNorm2d
        min_ft = 16
        conv_models = []
        conv_flat_dims = []
        self.use_affine_transform = use_affine_transform
        super().__init__(use_affine_transform, size, channels, norm_layer, ex_lat)

        for i in range(self.len_view):
            conv_mod = []
            conv_mod+=[DownConvModel(conv_kernels=[3,3], res_blocks=0, res_blocks_end=0,
                res_net=True, sampler=sampler[i], max_ft=ft_ch[i]*4, init_ft=ft_ch[i],
                target_flat_dim=target_flat_dim[i], img_size=size, channels = channels,
                norm_layer=norm_layer)
            ]
            fdim = batch_size*conv_mod[-1].flat_dim
            lat = latent_dim//2**i #less latent vector size for broader images
            hdim = min(fdim, ((fdim+batch_size*lat)//2))
            conv_mod += [
                nn.Flatten(0, -1),
                nn.Unflatten(0, (1, fdim)),
                nn.Linear(fdim, hdim),
                nn.GELU(),
                nn.Linear(hdim, lat),
                nn.GELU(),
            ]
            conv_models.append(nn.Sequential(*conv_mod))
            conv_flat_dims.append(lat)

        self.conv_models = nn.ModuleList(conv_models)

        
        if self.use_affine_transform:
            self.fc_ex = nn.Sequential(
                nn.Linear(ex_transform_dim, 32),nn.GELU(),
                *[nn.Linear(32,32), nn.GELU()]*3,
            )
        else:
            self.fc_ex = lambda x: x
            ex_lat = 0
        fc_map = [
            nn.Linear(sum(conv_flat_dims) + ex_lat*batch_size, latent_dim),
            nn.GELU(),
            *[nn.Linear(latent_dim, latent_dim), 
                nn.GELU()]*latent_layers
        ]
        self.fc_map = nn.Sequential(*fc_map)
        #self.localization = lambda x: x
        #self.fc_loc = lambda x: x

    def forward(self, values, **kwargs): 
        '''Values:tensor: B, V, C, W, H, batch, views, channels, width, height
            ----return tensor: B, L, batch, latent dim
        '''
        values = values.permute(1, 0, 2, 3, 4) #V, B, C, W, H
        values_ls = []
        if self.use_affine_transform:
            ex_values = kwargs['transform']
            ex_lat = self.fc_ex(ex_values)
            grid = self.st_grid(values[0], ex_lat.clone()) #B, lat(6)
        else:
            grid = None
        for i in range(self.len_view):
            iview = self.st_t(values[i], grid)
            ival = self.conv_models[i](iview) #B, C, W, H
            values_ls.append(ival)
        values = torch.cat(values_ls, dim = 1) #array -> tensor
        #values = values.view(self.batch_size, -1) #B, lat
        if self.use_affine_transform:
            values = torch.cat((values, ex_lat.view(1,-1)), dim = 1)
        values = self.fc_map(values)
        return values


class Latent2Retina(STRetina):
    def __init__(self, size = 64, views = [1,4,16], batch_size = 1,
        ft_ch = [64,32,16], latent_dim = 1024, sampler = [],
        target_flat_dim = [4096, 2048, 1024],
        channels=3, latent_layers = 3, ex_transform_dim = 5,
        use_affine_transform = False):
        ''' Model class for converting images from nviews to latent space
            it makes downsamplings then followed up by residual blocks, then
            flattening
            views: list: scales of the image
            size: int: size of the kernel filter
            batch_size: int:
            feature_channels: int: gets duplicated for each downsampling
            in_dim: list: input dimensions
            out_dim: list: output dimensions
            inter_dims: list: inter dimensions
            channels: int: RGB channels
            downsamplings: int:
            res_blocks: int:
        '''
        
        #print([batch_size,*in_dim, out_dim])
        
        self.batch_size = batch_size
        self.views = views
        self.len_view = len(views)
        self.size = size
        self.conv_dims = []
        self.channels = channels
        ex_lat = 32
        norm_layer = nn.InstanceNorm2d
        conv_models = []
        conv_flat_dims = []
        self.use_affine_transform = use_affine_transform
        super().__init__(use_affine_transform, size, channels, norm_layer, ex_lat)
        for i in range(self.len_view):
            conv_mod = []
            up_conv = UpConvModel(conv_kernels=[3,3], res_blocks=0, res_blocks_start=0,
                res_net=True, sampler=sampler[i], max_ft=ft_ch[i]*4, end_ft=ft_ch[i], 
                target_flat_dim=target_flat_dim[i], img_size=size, channels = 3,
                norm_layer=norm_layer)
            fdim = up_conv.flat_dim*batch_size
            lat = latent_dim//2**i
            hdim = min(fdim, ((fdim+batch_size*lat)//2))
            conv_mod += [
                nn.Linear(lat, hdim),
                nn.GELU(),
                nn.Linear(hdim, fdim),    
                nn.GELU(),
                nn.Unflatten(1, (batch_size, *up_conv.init_conv_dim)),
                nn.Flatten(0, 1),
                up_conv
            ]
            conv_models.append(nn.Sequential(*conv_mod))
            conv_flat_dims.append(lat)

        fc_map = []
        for _ in range(latent_layers):
            fc_map += [
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
            ]
        if self.use_affine_transform:
            pass
        else:
            ex_lat = 0
        self.ex_lat = ex_lat
        fc_map+=[
            nn.Linear(latent_dim, sum(conv_flat_dims) +  ex_lat*batch_size),
            nn.GELU(),
        ]
        self.conv_models = nn.ModuleList(conv_models)
        self.fc_map = nn.Sequential(*fc_map)
        self.conv_flat_dims = conv_flat_dims
        #self.localization = lambda x: x
        #self.fc_loc = lambda x: x

    def forward(self, values, **kwargs):
        ''' values:tensor: B, L, batch, latent dim
            ----return tensor: B, V, C, W, H, batch, views, channels, width, height
        '''
        values = self.fc_map(values) #B, lat
        ex_lat = values[:,:self.ex_lat*self.batch_size]
        ex_lat = ex_lat.view(self.batch_size, self.ex_lat)
        values = values[:,self.ex_lat*self.batch_size:]
        values_ls = []
        grid = None
        for i in range(self.len_view):
            fdim0 = (i>0)*self.conv_flat_dims[i-1]
            fdim1 = self.conv_flat_dims[i] + fdim0
            ival = values[:, fdim0:fdim1] #B, lat
            iview = self.conv_models[i](ival) #B, C, W, H
            if self.use_affine_transform:
                grid = self.st_grid(iview, ex_lat) if i==0 else grid
                iview = self.st_t(iview, grid)
            values_ls.append(iview)
        values = torch.stack(values_ls, 1) #B, V, C, W, H
        return values, {}

