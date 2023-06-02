from numpy import delete
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import get_io_slices, param_function, param_function_return
from generators.extra_models.utils import CellModelBase

class Emb(nn.Embedding):
    def forward(self, x, keys):
        return super().forward(keys)

def _get_eos_indeces(keys:torch.Tensor, batch_size: int):
    '''S, B, 1'''
    eos = (keys==2).nonzero(as_tuple=True)
    eos = eos[0][:batch_size]
    return eos

def _define_lstm_emb(vocab_size, emb_size, layers, lat, use_embedding = True, extra = 0):
    if use_embedding: 
        wemb = Emb(vocab_size, emb_size)
        lstm = nn.LSTM(emb_size + extra, lat, layers)
    else:
        wemb = lambda x, y: x
        lstm = nn.LSTM(vocab_size + extra, lat, layers)
    return wemb, lstm

class Keys2Latent(CellModelBase):
    def __init__(self, batch_size, lat, device, layers = 2,
            vocab_size = 1, use_embedding=False, emb_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.use_embedding= use_embedding
        self.lstmout = torch.tensor(0)

        self.wemb, self.lstm = _define_lstm_emb(
            vocab_size, emb_size, layers, lat//2, use_embedding,
        )
        big_lat = lat*batch_size*layers #2*lat//2, cancels out c0 and h0
        mid_lat = (big_lat + lat)//2
        self.fc_h0 = nn.Sequential( #h0 -> lat
            #nn.Linear(layers*alat, lat), nn.GELU()
            nn.Linear(big_lat, mid_lat), nn.GELU(),
            nn.Linear(mid_lat, lat), nn.GELU(),
        )
        self.hsize = (layers, batch_size, lat//2)
        self.h0 = torch.rand(self.hsize, device = device)
        self.c0 = torch.rand(self.hsize, device = device)
        self.out_dim = lat
    def reset_hidden(self, ):
        self.h0 = torch.zeros_like(self.h0)#self.h0.detach_()
        self.c0 = torch.zeros_like(self.c0)#self.c0.detach_()

    def core_forward(self, x:torch.Tensor):
        self.reset_hidden()
        keys = torch.argmax(x.detach(), dim=2)
        emb = self.wemb(x, keys)
        out, (h0, c0)= self.lstm(
            emb, (self.h0, self.c0)) #S, Lat
        self.h0, self.c0 = h0, c0
        return out, h0, c0

    def forward(self, x:torch.Tensor, **kwargs: dict):
        ''' x: S,B,V
        '''
        _, h0, c0 = self.core_forward(x)
        hc0 = [h0.view(1,-1), c0.view(1,-1)]
        out = self.fc_h0(torch.cat(hc0, dim=1)) #1, Lat
        return out

class Mouse2Latent(Keys2Latent):
    def __init__(self, batch_size:int, lat:int, 
            device:torch.device, layers: int,
            dim_param:dict, param_names: list,
            lat_param:dict):
        self.inlat = sum(lat_param.values())
        self.indim = sum(dim_param.values())
        super().__init__(batch_size=batch_size, 
            lat=lat, device=device, layers=layers,
            vocab_size=self.inlat, use_embedding=False)
        self.param_names = param_names
        self.slices = get_io_slices([dim_param, lat_param], param_names)
        self.lat_fc = nn.Sequential(
            nn.Linear(sum(dim_param.values()), self.inlat),
            nn.GELU(),
            *[nn.Sequential(
                    nn.Linear(self.inlat, self.inlat), nn.GELU()
                ) for x in range(1)],
        )
    def in2lat(self, parameters, seq_len):
        in_val = torch.zeros(
            (seq_len, self.batch_size, self.indim),
            device=self.device
        )
        #in_val = nn.Parameter(in_val)
        for pn in self.param_names:
            to_add = in_val[:, :, self.slices[pn][0]]
            in_val[:, :, self.slices[pn][0]] = to_add + parameters[pn]
        return self.lat_fc(in_val) # S, B, vocab

    def forward(self, x:torch.Tensor, **kwargs:dict):
        ''' x: S,B,V
        '''
        kwargs['cmd'] = x
        in_lats = self.in2lat(kwargs, x.size()[0]) # S, B, vocab
        _, h0, c0 =  self.core_forward(in_lats)
        hc0 = [h0.view(1,-1), c0.view(1,-1)]
        out = self.fc_h0(torch.cat(hc0, dim=1)) #1, Lat
        return out

class Latent2Keys(Keys2Latent):
    def __init__(self, batch_size, lat, device, layers = 2,
            vocab_size = 1, use_embedding=False, emb_size=1, superinit = True):
        if superinit:
            super().__init__(batch_size, lat, device, layers,
                vocab_size, use_embedding, emb_size)
        #self.lsfmax = nn.LogSoftmax(dim = 2)
        self.vocab_size = vocab_size
        self.kfc_o = nn.Sequential(
            nn.Linear(lat//2, vocab_size),
        )
        big_lat = lat*batch_size*layers
        mid_lat = (big_lat + lat)//2
        self.fc_ch0 = nn.Sequential(
            nn.Linear(lat, mid_lat), nn.GELU(),
            nn.Linear(mid_lat, big_lat), nn.GELU(),
        )

    def forward(self, x: torch.Tensor, **kwargs:dict):
        ''' 1, Lat
        '''
        keyval = kwargs['keys'].detach()
        self.h0, self.c0 = [
            h.view(self.hsize) for h in self.fc_ch0(x).chunk(2, dim=1)
        ] #Layers, B, Lat
        out, _, _ = self.core_forward(keyval)
        out = self.kfc_o(out)
        return out, {} # seq, batch, vocab

class Latent2Mouse(Mouse2Latent, Latent2Keys):
    def __init__(self, batch_size, lat, device, layers = 2, param_names = [],
            dim_param: dict = None, lat_param: dict = None):
        Mouse2Latent.__init__(self, batch_size=batch_size, lat=lat,
            device=device, param_names= param_names,
            layers=layers, dim_param=dim_param, lat_param=lat_param)
        Latent2Keys.__init__(self, batch_size=batch_size, lat=lat,
            device=device, layers=layers, vocab_size=self.indim, 
            superinit=False)

        self.maturity = 0.0
        self.fc_l2il = nn.Sequential(
            nn.Linear(lat//2, self.inlat), nn.GELU(),
        )
        self.kfc_o = nn.Sequential(
            *[nn.Sequential(
                    nn.Linear(self.inlat, self.inlat), nn.GELU()
                ) for x in range(1)],
            nn.Linear(self.inlat, self.indim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, **kwargs:dict):
        ''' 1, Lat
        '''
        keyval = self.in2lat(kwargs, kwargs['cmd'].size()[0])
        self.h0, self.c0 = [
            h.view(self.hsize) for h in self.fc_ch0(x).chunk(2, dim=1)
        ] #Layers, B, Lat
        lat, _, _ = self.core_forward(keyval)
        out = self.fc_l2il(lat)
        out = self.kfc_o(out)
        pnout = {}
        for pn in self.param_names:
            pnout[pn] = out[:, :, self.slices[pn][0]]
        mainout = pnout['cmd']
        del(pnout['cmd'])
        return mainout, pnout # seq, batch, vocab
