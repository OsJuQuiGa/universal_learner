from sd_patch import sd
import numpy as np
import torch
from scipy import signal
from scipy.io.wavfile import write 
import torchvision.transforms.functional as Ft
from terminals.generic import Terminal, OutputTerminal
import multiprocessing as mp
import threading as tr
import os
import time
from torchaudio.transforms import Resample

class SystemAudio(Terminal):
    '''Audio is highly compressed with layers with kernel filters up to 18
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.batch_size = 1
        self.io_channels = 2
        self.downsamplings = 2
        self.sampling = 40000
        self.fs = 44100
        self.d = 2
        self.rate = settings['terminal_sockets']['SystemAudio']['rate']
        self.featchannels = settings['terminal_sockets']['SystemAudio']['channels']
        self.sddim = [
            self.rate,
            self.batch_size, 
            self.io_channels,
            self.fs*self.sampling,
        ]
        self.parameters_settings.update({
            'audio':{
                'dimensions': self.sddim,
                'tensortype': 'float',
                'values': torch.rand(self.sddim, 
                    device = self.device)*0.2 - 0.1,
                'model': None,
                'latent_dim': 512,
                'model_settings': {
                    'batch_size':self.batch_size,
                    'channels':self.io_channels,
                    'd': self.d,
                    'fs': self.sampling,
                    'feature_channels': self.featchannels,
                },
            },
        })
        self.param_names+=['audio']
        self.subaction_settings['internn']*=2
        self.has_real_expression = True

    def save_sample(self, audio=None):
        sample_path = os.path.join('./samples', self.category)
        if audio == None:
            audio = self.parameters['audio']
        samples = [audio]
        names = ['orig']
        for i in range(len(samples)):
            sam = samples[i].squeeze(dim=0).t()
            sam = (sam.detach().cpu().numpy()*2**15).astype(np.int16)
            sname = sample_path + '_'+ str(names[i]) + '.wav'
            write(sname, self.fs, sam)

class SystemAudioInput(SystemAudio):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['audio']['model'] = 'Sound2Latent'

class SystemAudioInputCritic(SystemAudioInput):
    def __init__(self, settings):
        super().__init__(settings)
        self.last_time = time.time()
        self.r_t = 5
        self.data = self.rec_data()
        audio_sett = self.parameters_settings['audio']
        audio_sett['method_fn'] = self.get_record
        audio_sett['model_settings']['notediv'] = 12

    def rec_data(self):
        '''records audio sample of the time in terminal.r_t
        '''
        data = sd.rec(
            frames= self.fs*self.r_t, samplerate=self.fs, 
            channels = self.io_channels,
        )
        return data

    def get_record(self, Hmemory, phase):
        '''gets the record and process it in tensor form, the past
            record is stopped early if it hasn't finnished
        '''
        audio_data = self.data
        self.data = self.rec_data()
        now = time.time()
        thres = now - self.last_time
        if thres < self.d:
            slc = slice(None, self.fs*self.d)
        else:
            int_t = int(min(thres, self.r_t)*self.fs)
            slc = slice(int_t-self.fs*self.d, int_t)
        media = torch.from_numpy(audio_data).to(
            self.device).float()[slc,:].permute(
                1,0).unsqueeze(dim=0).clip(-2.0,2.0)
        media[torch.isnan(media)] = 0.0
        #self.save_sample(media)
        return media

class SystemAudioOutput(SystemAudio):
    '''the audio uses entropy loss so it can produce more diverse results
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.batch_size = 1
        self.parameters_settings['system_audio']['model'] = 'Latent2Sound'
        self.parameters_settings['system_audio']['entropy'] = False

class SystemAudioOutputReal(SystemAudioOutput):
    '''creates a stream that is independed of the recording
        that can play sounds
    '''
    def express_values(self, parameters):
        '''converts values to audio data
        '''
        audio_data = parameters['system_audio'].cpu(
            ).detach().squeeze(dim=0).permute(1,0).contiguous().numpy()
        source = self.category_dict['source']
        if self.settings['terminal_sockets'][source]['express']:
            sd.play(audio_data)





















