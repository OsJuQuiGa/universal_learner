import os
import math
import threading
import Xlib.threaded
import keyboard
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from terminals.media.periferical import write_key_sequence
from utils import check_key, get_bound_loss_fn, time_encoding, _time_dict_denom
from terminals.generic import Terminal, OutputTerminal

text_keys = ['!', '"', '#', '$', '%', '&', '\'', '(',
')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
'8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
'a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

cmd_keys = [
'altleft', 'backspace', 'capslock',
'ctrlleft', 'delete', 'down', 'end', 'enter', 'esc', 
'f1', 'f10', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
'pagedown', 'pageup', 'right', 'left', 'shiftleft', 'space', 'tab', 
'up', 'printscreen'
#'f11', 'f12' #f11 blocks mouse, f12, blocks 
#'hanguel', 'hangul', 'hanja', 'home', 'insert',
#'kana', 'kanji',
#'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
#'num7', 'num8', 'num9', 
]
keyboard_keys = [*text_keys, *cmd_keys]
cmds_fn = {
    'down':pyautogui.keyDown, 
    'up':pyautogui.keyUp, 
    #'press':pyautogui.press,
}
all_key_tokens = []
for k in keyboard_keys:
    for j in cmds_fn.keys():
        all_key_tokens.append(' '.join([k, j]))

all_key_tokens = ['null', 'SOS', 'EOS', *all_key_tokens]

class SystemKeyboard(Terminal):
    ''' Still not tested
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.use_emd = True
        self.vocab_size = len(all_key_tokens)
        self.embedding_dim = 100
        self.use_emd = True
        self.rate = settings['terminal_sockets']['SystemKeyboard']['rate']
        self.seq_len = self.rate * 2
        keys_t = torch.zeros((self.seq_len, 
            self.batch_size, 
            self.vocab_size), device=self.device)
        keys_t[0] = 1.0 #EOS
        d_t = torch.zeros((self.seq_len, 
            self.batch_size, 
            1), device=self.device)
        d_t[0] = 1.0
        self.parameters_settings.update({
            'keys':{
                'dimensions': (
                    self.seq_len, 
                    self.batch_size, 
                    self.vocab_size),
                'tensortype': 'float',
                'values': keys_t,
                'model': None,
                'latent_dim': 256,
                'entropy': False,
                'model_settings': {
                    'batch_size': self.batch_size,
                    'vocab_size': self.vocab_size,
                    'use_embedding': self.use_emd,
                    'emb_size': self.embedding_dim,
                    'layers':2,
                },
                'sequential': 0,
                'categorical': True,
            },
        })
        self.param_names +=['keys']

class SystemKeyboardOutput(SystemKeyboard):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['keys']['model'] = 'Keys2Latent'


class SystemKeyboardOutputCritic(SystemKeyboardOutput):
    def __init__(self, settings):
        super().__init__(settings)


class SystemKeyboardOutputReal(SystemKeyboardOutputCritic):
    def __init__(self, settings):
        super().__init__(settings)
        ''' must be batch 1, one series of actions at the time
        '''
        self.press_keep_timeout = 4 #max cycles allowed to be down
        self.keys_down_cycles = {x: 0 for x in keyboard_keys}
        self.parameters_settings['keys']['dimensions'] = (
            self.seq_len, 
            self.batch_size, 
            self.vocab_size)
        self.parameters_settings['keys'][
            'model_settings']['batch_size'] = self.batch_size
        self.parameters_settings.update({
            'keys_category':{
                'dimensions': (
                    self.seq_len, 
                    self.batch_size),
                'tensortype': 'long',
                'values': 1,
                'model': 'keys',
                'latent_dim': 256,
                'entropy': False,
                'model_settings': {},
                'sequential': -1,
                'categorical': False,
                'loss_fn': None #no training, only placeholder
            },
        })

    def express_values(self, parameters):
        #the values before softmax are the ones tentatively save
        #set in Hmemory
        super().express_values(parameters)
        keys = parameters['keys']
        kdd = self.keys_down_cycles
        for bindx in range(self.batch_size):
            sequence_keys = []
            keys_to_up = []
            keyindeces = torch.argmax(keys[:,bindx], dim=1) #argmax sequence
            for sindx in range(self.seq_len):
                keyindx = keyindeces[sindx].item()
                keystr = all_key_tokens[keyindx]
                key, cmd = keystr.split(' ')
                pdb = cmd == 'down'
                kdd[key] *= pdb
                kdd[key] += (0 == kdd[key]) * pdb
                sequence_keys.append((
                    key, cmd
                ))
            for k in kdd.values(): #checks how many cycles some
                #keys has been down.
                kdd[k] += kdd[k] > 0
                if kdd[k] > self.press_keep_timeout:
                    keys_to_up.append(k)
                    kdd[k] = 0

            for k in keys_to_up:
                sequence_keys.insert(1, (
                    k, 'press_up'
                ))
            delayed_mup = threading.Thread(
                target=write_key_sequence, args={sequence_keys, cmds_fn})
            delayed_mup.daemon = True
            delayed_mup.start()
        

class SystemKeyboardInput(SystemKeyboard):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['keys']['model'] = 'Latent2Keys'

class SystemKeyboardInputCritic(SystemKeyboardInput):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['keys']['method_fn'] = self.record_keystrokes
        #self.keys_up = []
        self.keyboard = keyboard
        self.keyboard.start_recording()
        self.last_t = time.time() - 1
        self.time_seq = None


    def record_keystrokes(self, Hmemory, phase):
        key_events = [keyboard.stop_recording()] #sources of keystrokes
        keyseq_z = torch.zeros_like(self.parameters['keys'])
        keys = self.parameters['keys']
        keys[0][:, 0] =  1.0 #SOS 
        time_item = 1/self.rate #constant time for each item in sequence
        for b in range(self.batch_size):
            b_events = key_events[b]
            init_time = b_events[0].time
            last_time = b_events[-1].time
            diff_time = last_time - init_time
            seq_len = 2 + diff_time//time_item #2+ for SOS and EOS
            seq_diff = max(0, seq_len - keys.size()[0])
            if seq_diff > 0:
                pdim = self.parameters_settings['keys']['dimensions'][2]
                eten = torch.zeros((seq_diff, self.batch_size,pdim),
                    device = self.device)
                self.parameters['keys'] = torch.cat(keys, eten)
                keys = self.parameters['keys']
            keys[0][b][0] = 1.0
            seq_time = np.linspace(last_time, init_time, seq_len - 2)
            button_queue = []
            for event in b_events:
                etime = event.time - init_time
                ccond = False
                tdx = 0
                while(etime < seq_time[tdx+1] or etime > seq_time[tdx]):
                    if len(button_queue) > 0:
                        be = button_queue.pop(0)
                        if be.name in keyboard_keys:
                            str2indx = ' '.join(be.event_type, be.name)
                            indx = all_key_tokens.index(str2indx)
                        else: indx = 2
                    else: indx = 2
                    sidx = tdx + 1
                    keyseq_z[sidx][b][indx] = 1.0
                    tdx += 1
                    if tdx == len(seq_time):
                        ccond = True
                if ccond:
                    keyseq_z[-1][b][1] = 1.0
                    break
                button_queue.append(event)
        keyboard.start_recording()
        return keyseq_z