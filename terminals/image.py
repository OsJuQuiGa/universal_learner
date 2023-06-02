import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import multiprocessing
import Xlib.threaded
from utils import (foveate,
        affn_limit_presets, limit_fn, gaussian_policy_loss,
        gaussstdvar_loss
    )

from terminals.media.screenshot import (Screenshot,
    get_monitor, range_dim, save_tensor_image)
from terminals.generic import Terminal


class Retina(Terminal):
    '''Retina that takes a glimpse in a x,y coordinate composed of
    3(more or less) scales of the image
    '''
    def __init__(self, settings):
        super().__init__(settings) #fix mainfolder, use instead class definition
        ret_settings = settings['terminal_sockets']['Retina']
        self.views = ret_settings['views']
        self.featchannels = ret_settings['featchannels']
        self.size = ret_settings['size']
        self.channels = ret_settings['channels']
        self.affn_elements = ret_settings['affn_elements']
        monitor, big_monitor = get_monitor(settings['screens']['slave'][0])

        keys = settings['screen_dims']
        self.screen_range = torch.stack([
            range_dim(torch.tensor(big_monitor[key], monitor[key])) 
            for key in keys
        ]).t()
        t_tensor = torch.stack(torch.tensor(
            [0.0, 1.0, 1.0, 0.0, 0.0]*self.batch_size), dim=0) #B, T

        nimage_dim = (
            self.batch_size,
            len(self.views),
            self.channels,
            self.size, 
            self.size,
        )
        #self.hall_op = torch.sum
        self.parameters_settings.update({
            'transform':{ #Rotation(-1,1), ,ScaleX, ScaleY(2.0,0.5), LocX, LocY(-1,1)
                'dimensions': t_tensor.size(),
                'tensortype': 'float',
                'values': t_tensor,
                'model': 'CellMLP',
                'latent_dim': 32,
                'model_settings': {
                    'layers':3,
                },
                'sequential': -1,
                'categorical': False,
            },
            'nimage':{
                'dimensions': nimage_dim,
                'tensortype': 'float',
                'values': torch.rand(nimage_dim)*0.2 - 0.1,
                'latent_dim': 512,
                'model': None,
                'model_settings': {
                    'views': self.views,
                    'size': self.size,
                    'channels': self.channels,
                    'batch_size': self.batch_size,
                    'ft_ch': self.featchannels,
                    'target_flat_dim': ret_settings['target_flat_dim'],
                    'use_affine_transform': ret_settings['use_affine_transform']
                },
                'sequential': -1,
                'categorical': False,
            },
        })
        self.param_names+=['transform', 'nimage']
        #self.subaction_settings['internn']*=2

    def save_views(self, nimage = None):
        '''saves a glimpse with its scales(views)'''
        for i in range(len(self.views)):
            if nimage == None:
                view = self.parameters['nimage'][:,i,:,:,:]
            else:
                view = nimage[:,i,:,:,:]
            save_tensor_image(view[0], self.category+'_'+str(i)) 

class RetinaInput(Retina):
    '''Input retina uses the Retina2Latent model'''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['nimage']['model'] = 'Retina2LatentV2'
        self.parameters_settings['transform']['model'] = 'nimage'

class RetinaInputEncoder(RetinaInput):
    '''Defines a Screenshot object that is ready
    to take shots of the screen when prompted, also sets the method_fn
    for the nimage parameter that holds the glimpse'''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['nimage']['method_fn'] = self.take_glimpse
        self.screenshot = Screenshot(
            settings['screens'][0], self.channels
        )

    def take_glimpse(self, Hmemory, phase):
        '''takes a screenshot of the screen and makes the glimpse(nviews)
            Hmemory: Hmemory_obj
            phase: dict
            ----return nviews: tensor((batch, nviews, y, x, ch)-dim)
        '''
        self.screenshot.take()
        media = self.screenshot.get_pytorch_img(self.device)
        if 'transform' in self.param_names:
            transform = self.parameters['transform']
            rot = limit_fn(transform[:,0], affn_limit_presets['extrinsic']['rotation'])
            scale = limit_fn(transform[:,1:2] + 1.0, affn_limit_presets['extrinsic']['scale']) #+1
            loc = limit_fn(transform[:,3:4], affn_limit_presets['extrinsic']['loc']) #0 to 1.0
            media = TF.resize(media, scale)
            media = TF.rotate(media, rot, center = (loc+1)/2)

        sr = self.screen_range
        l_ten = loc*sr[1] + sr[0] # x * m + b
        nviews = foveate(media, l_ten, self.size, self.views)
        #media.to(device=self.device)
        #media += self.parameters['nimage']*self.parameters['hallucination']
        return nviews

class RetinaOutput(Retina):
    '''Retina Output uses Latent2Retina model for the nimage parameter'''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['nimage']['model'] = 'Latent2RetinaV2'

class RetinaOutputCritic(RetinaOutput):
    ''' Retina output critic part'''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['nimage']['loss_fn'] = F.mse_loss
        self.parameters_settings['transform']['loss_fn'] = F.mse_loss

class RetinaOutputReal(RetinaOutput):
    '''Retina output real doesn't use nimage, in this case it
        would be reserved for a canvas program that allows this expression
        to be draw, and only left with reward and l parameters
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['transform']['loss_fn'] = gaussian_policy_loss
        self.parameters_settings['transform']['variance_loss_fn'] = gaussstdvar_loss
        self.param_names.remove('nimage')
        #self.parameter_settings['hallucination'] = self.hall_parameter_settings



        
        



        
    
    
