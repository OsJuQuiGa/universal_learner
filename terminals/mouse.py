import multiprocessing
import threading
import Xlib.threaded
import mouse
import torch
import torch.nn as nn
import pyautogui
import time
import numpy as np
from utils import check_key, get_bound_loss_fn, get_bound_loss_unbounded_fn
from terminals.generic import Terminal
from terminals.media.screenshot import get_monitor, range_dim

mouse_keys = [
    'left', 'middle', 'right'
]
cmds_fn = {
    'down':pyautogui.mouseDown,
    'up':pyautogui.mouseUp,
}
all_key_tokens = []
for k in mouse_keys:
    for j in cmds_fn.keys():
        all_key_tokens.append(' '.join([k,j]))
#n_scrolls = 3
#extra_cmd = {'scroll_up':lambda x: pyautogui.scroll,
#            'scroll_down':lambda x: pyautogui.scroll,}
#for i in range(n_scrolls):
#    for j in extra_cmd.keys:
#        all_key_tokens.append(' '.join(j, i))
#        all_key_tokens.append(' '.join(j, -i))
#cmds_fn.update(extra_cmd)
all_key_tokens = ['null ', 'SOS ', 'EOS ', *all_key_tokens]

class SystemPointer(Terminal):
    ''' Still not tested
    '''
    def __init__(self, settings):
        #Mainfolder: Folder where the data of this terminal is stored
        #Monitor:Which monitor the pointer is going to navigate
        super().__init__(settings)
        monitor, big_monitor = get_monitor(settings['screens']['slave'][0])
        keys = settings['screen_dims']
        self.batch_size = 1
        self.screen_range = torch.stack([
            range_dim(torch.tensor(big_monitor[key], monitor[key]), device = self.device) 
            for key in keys
        ]).t()
        self.monitor = monitor
        self.big_monitor = big_monitor
        self.rate = settings['terminal_sockets']['SystemPointer']['rate']
        self.seq_len = self.rate * 2
        self.scroll_delta = 4
        self.vocab_size = len(all_key_tokens)
        l_dim = (self.seq_len, self.batch_size, 2)
        self.parameters_settings.update({
            'cmd':{ 
                'dimensions': (
                    self.seq_len, self.batch_size, 
                    self.vocab_size),
                'tensortype': 'float',
                'values': None,
                'latent_dim':[32],
                'model': None, #'MouseSeq2Latent'
                'model_settings': {
                    'layers':2,
                },        
            },
            'scroll':{
                'dimensions': (
                    self.seq_len, self.batch_size, 
                    1),
                'tensortype': 'float',
                'values': None,
                'model': 'cmd',
            },
            'l':{ #-1.0,1.0
                'dimensions': l_dim,
                'tensortype': 'float',
                'values': None,
                'model': 'cmd',
                'latent_dim':32,
                'model_settings': {
                    'layers':2,
                },
            },
            'pressure':{ #if graphics tablet is connected
                'dimensions': (
                self.seq_len, self.batch_size, 
                1),
                'tensortype': 'float',
                'values': None,
                'model': 'cmd',
            },
        })
        self.param_names += ['cmd', 'l', 'scroll']

class SystemPointerInput(SystemPointer):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['cmd']['model'] = 'Mouse2Latent'

class SystemPointerInputCritic(SystemPointerInput):
    def __init__(self, settings):
        super().__init__(settings)
        def tmp(name):
            self.parameters_settings[name]['method_fn'] = self.generic_method_fn(name)
        self._param_function(p_func=tmp)
        self.parameters_settings['cmd']['method_fn'] = self.get_system_pointer
        self.mouse = mouse
        self.events = []
        self.mouse.hook(self.events.append)
        self.last_time = time.time() - 1
        self.last_moves_t = [torch.zeros(
            self.parameters_settings['l']['dimensions'], self.device)]*self.batch_size

    def get_media_pointer(self, Hmemory, phase):
        sr = self.screen_range
        self.mouse.unhook(events.append)
        events = [self.events]
        self.events = []
        self.mouse.hook(self.events.append)
        def tmp(name):
            self.parameters[name] = torch.zeros(
                self.parameters_settings[name]['dimensions'], 
                device=self.device) #resets input tensor and the dimension
        self.parameters['cmd'][0][:, 0] =  1.0 #SOS 
        self._param_function(self.param_names, tmp)
        time_item = 1/self.rate #constant time for each item in sequence
        for b in range(self.batch_size):
            b_events = events[b]
            init_time = b_events[0].time
            last_time = b_events[-1].time
            diff_time = last_time - init_time
            seq_len = 2 + diff_time//time_item #2+ for SOS and EOS
            def tmp(name):
                seq_diff = max(0, seq_len - self.parameters[name].size()[0])
                if seq_diff > 0:
                    pdim = self.parameters_settings[name]['dimensions'][2]
                    eten = torch.zeros((seq_diff, self.batch_size, pdim),
                        device=self.device)
                    self.parameters[name] = torch.cat([self.parameters[name], eten]) #makes the parameters with longer sequences
            self._param_function(self.param_names, tmp)
            self.parameters['cmd'][0][b][0] = 1.0 #SOS
            self.parameters['scroll'][0][b] = 0.0
            self.parameters['l'][0][b] = self.last_moves_t[b]
            seq_time = np.linspace(last_time, init_time, seq_len - 2) #2- for SOS and EOS
            #real seq len to be taken
            last_move = (0.0,0.0)
            button_queue = []
            scroll_delta = 0
            ccond = False
            tdx = 0
            for event in b_events: #time of each element
                etime = event.time - init_time
                ccond = False
                while(etime < seq_time[tdx + 1] or etime > seq_time[tdx]):
                    #changes index in sequence
                    #button processing
                    if len(button_queue)>0:
                        be = button_queue.pop(0)
                        str2indx = ' '.join(be.event_type, be.button)
                        indx = all_key_tokens.index(str2indx)
                    else:
                        indx = 2
                    sidx = tdx + 1
                    self.parameters['cmd'][sidx][b][indx] = 1.0
                    #location processing
                    l_ten = torch.tensor(last_move, device=self.device)
                    l_ten = (l_ten*sr[1] + sr[0])*2 - 1
                    self.parameters['l'][sidx][b] = l_ten
                    #scroll processing
                    self.parameters['scroll'][sidx][b] = scroll_delta
                    scroll_delta = 0
                    tdx += 1
                    if tdx == len(seq_time):
                        ccond = True
                if ccond: 
                    self.parameters['cmd'][-1][b][1] = 1.0
                    self.parameters['l'][-1][b] = l_ten
                    self.parameters['scroll'][-1][b] = 0.0
                    break
                if type(event) == mouse._mouse_event.MoveEvent:
                    last_move = (event.y, event.x)
                elif type(event) == mouse._mouse_event.ButtonEvent:
                    button_queue.append(event)
                elif type(event) == mouse._mouse_event.WheelEvent:
                    scroll_delta += event.delta
            self.last_moves_t[b] = l_ten
        return self.parameters['cmd']

class SystemPointerOutput(SystemPointer):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['mousecmd']['model'] = 'Latent2Mouse'


class SystemPointerOutputReal(SystemPointerOutput):

    def __init__(self, settings):
        super().__init__(settings)
        self.press_keep_timeout = 10
        self.batch_size = 1
        self.keys_down_cycles = {x:0 for x in mouse_keys}
        self.parameters_settings['cmd']['dimensions'] = (
            self.seq_len, 
            self.batch_size, 
            self.vocab_size)
        self.parameters_settings['cmd'][
            'model_settings']['batch_size'] = self.batch_size
        self.keys_to_up = [[]*self.batch_size]
        self.big_monitor_t = (self.big_monitor['height'], self.big_monitor['width'])
        self.big_monitor_t = torch.tensor(self.big_monitor_t, device=self.device)
        
    def express_values(self, parameters):
        #gets the real input
        super().express_values(parameters)
        sr = self.screen_range
        keys = parameters['cmd'].detach().clone()
        scroll_seq = parameters['scroll'].detach().clone()
        l_seq = parameters['l'].detach().clone().clip(-1.1,1.05)
        l_seq = (l_seq + 1.0)/2.0
        #l_seq = (l_seq - sr[0])/sr[1]
        kdd = self.keys_down_cycles
        for bindx in range(self.batch_size):
            sequence_events = []
            first_l = None
            keyindeces = torch.argmax(keys[:,bindx], dim=1) #argmax sequence
            for sindx in range(self.seq_len):
                keyindx = keyindeces[sindx].item()
                keystr = all_key_tokens[keyindx]
                key, cmd = keystr.split(' ')
                pdb = cmd == 'down'
                kdd[key] *= pdb
                kdd[key] += (0 == kdd[key]) * pdb
                l = l_seq[sindx][bindx]
                l = l*sr[1] + sr[0]
                l = torch.round(l*self.big_monitor_t)
                l = l.tolist()
                if first_l is None:
                    first_l = l
                scroll = scroll_seq[sindx][bindx].item()
                sequence_events.append((
                    key, cmd, l, scroll
                ))
            for k in kdd.values():
                kdd[k] += kdd[k] > 0
                if kdd[k] > self.press_keep_timeout:
                    self.keys_to_up.append(k)
                    kdd[k] = 0
            for k in self.keys_to_up:
                sequence_events.insert(1, (
                    k, 'up', first_l, 0
                ))
            
        return parameters
