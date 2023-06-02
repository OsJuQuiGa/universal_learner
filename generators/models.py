from statistics import mean
import time
import math
from collections import OrderedDict
import torch
from torch import nn, Tensor
from generators.extra_models.utils import CellModelBase, CellModelWithParameters
from generators.image import Retina2Latent, Latent2Retina
from generators.sound import Sound2Latent, Latent2Sound
from utils import dynamic_average_pyt, get_io_slices, param_function, param_function_return
#from generators.extra_models.transformers import AttnTransformer

#from typing import Optional, Any
#from torch import Tensor

class NULL(nn.Module):
    ''' A dummy layer used for returning a constant tensor
        or return the tensor that is put into 
    '''
    def __init__(self, *args, constant_tensor=None, **kwargs):
        super().__init__()
        if constant_tensor:
            self.main = lambda x: constant_tensor
        else:
            self.main = lambda x: x
    def forward(self, x):
        return self.main(x)

class RelativeModulator(nn.Module):
    ''' Neuromodulator that uses the mean, standarization or nothing
        of an sequence to back prop
    '''
    def __init__(self, lat, singular_modulation=False):
        super().__init__()
        layers = 3
        model = [nn.Linear(lat, lat), nn.GELU()]*layers
        if singular_modulation:
            model+=[nn.Linear(lat, 1)]
        self.fc = nn.Sequential(*model)
        self.act = nn.Tanh()

    def forward(self, lat_bk, seq):
        '''seq: seq, B, lat'''
        mean_t = torch.mean(seq, dim=0) #B, lat
        std = torch.maximum(0.1, torch.std(seq, dim=0)) #B, lat
        modulator = self.act(self.fc(lat_bk)) #B, (lat or 1)

        abs_mod = torch.relu(modulator)
        rel_mod = 1 - torch.abs(modulator)
        std_mod = torch.relu(1 - modulator)

        absolute = seq
        relative = seq - mean_t
        standard = relative/std

        output = (abs_mod*absolute + rel_mod*relative 
            + std_mod*standard) #B, seq, lat
        return output

class MLP(nn.Module):
    '''Multi-Layer perceptron model
    layers = number of layers
    dims = if len() 1, all layer have the same dimensions, if len() 2
    then indx0 is the input dim, indx1 is the output dim, and the rest is
    the average of the two, if len() 3, the same as 2 but the rest is the
    dimension of indx 1, else len()> 3, must have the same len()
    as layers and speciefies the dim of all layers
    activation = activation layer of nn module
    '''

    def __init__(self, layers = 3, in_dim= 256, out_dim = None,
            inter_dims:list = None, activation = nn.GELU(), batch_size = 1,
        ):
        '''layers: int: number of layers in MLP
            in_dim: int:
            out_dim: int:
            inter_dims: list:
            activation: func:
            batch_size: int
        '''
        super().__init__()
        model = []
        if out_dim is out_dim:
            out_dim = in_dim
        if inter_dims is None:
            inter_dims = []
            
        self.in_dim = (batch_size, in_dim)
        self.out_dim = (batch_size, out_dim)
        
        dims = [self.in_dim[1], *inter_dims, self.out_dim[1]]
        #print(dims)
        if len(dims) == 0 or layers == 0:
            layers = 0
            model = lambda x: x
        elif len(dims) == 1:
            dimensions = [dims[0] for x in range(layers+1)]
        elif len(dims) == 2:
            avg = (dims[0] + dims[1])/2
            dimensions = [avg for x in range(layers+1)]
            dimensions[0], dimensions[-1] = dims[0], dims[1]
        elif len(dims) == 3:
            dimensions = [dims[1] for x in range(layers+1)]
            dimensions[0], dimensions[-1] = dims[0], dims[2]
        elif len(dims) > 3:
            assert layers + 1 == len(dims)
            dimensions = dims
        for i in range(layers):
            in_dim = int(dimensions[i])
            out_dim = int(dimensions[i+1])

            lin_mod = nn.Linear(in_features=in_dim, out_features=out_dim)
            #nn.init.xavier_uniform_(lin_mod.weight, gain=1e-1)
            nn.init.uniform_(lin_mod.weight, -1e-1, 1e-1)
            nn.init.uniform_(lin_mod.bias, -1e-1, 1e-1)
            model.append(lin_mod)

            #model += [nn.LayerNorm(out_dim, elementwise_affine=False)]

            model.append(activation)
        self.model = nn.Sequential(*model)

    def forward(self, values):
        to_return = self.model(values)
        return to_return

class CellMLP(CellModelBase):
    def __init__(self, layers=3, in_dim=256, 
            out_dim=None, inter_dims=[], activation=nn.GELU(),
            batch_size=1, reverse=False, **kwargs):
        super().__init__()
        if reverse:
            p_out_dim = out_dim
            out_dim = in_dim
            in_dim = out_dim if p_out_dim is None else p_out_dim
            inter_dims = tuple(reversed(inter_dims))
        self.model = MLP(layers=layers, in_dim=in_dim, out_dim=out_dim,
            inter_dims=inter_dims, activation= activation,
            batch_size=batch_size)
        self.out_dim = self.model.out_dim[1]
    def forward(self, values):
        return self.model(values)

class InputMLP(CellMLP):
    def forward(self, values, **kwargs):
        return self.model(values)

class OutputMLP(CellMLP):
    def forward(self, values, **kwargs):
        return self.model(values), {}

class InputParamMLP(CellModelWithParameters):
    def __init__(self, layers=3, in_dim=256, 
            out_dim=None, inter_dims=[], activation=nn.GELU(),
            batch_size=1, dim_param={}, param_names=tuple(), lat_param={}, **kwargs):
        super().__init__(dim_param, param_names, lat_param)
        self.mlp = CellMLP(layers=layers, in_dim=in_dim, out_dim=out_dim,
            inter_dims=inter_dims, activation= activation, batch_size=batch_size)
        self.batch_size = batch_size

    def forward(self, values, **kwargs):
        kwargs[self.param_names[0]] = values
        in_lats = self.in2lat(kwargs, (self.batch_size,), values.device)
        return in_lats

class OutputParamMLP(CellModelWithParameters):
    def __init__(self, layers=3, in_dim=256, 
            out_dim=None, inter_dims=[], activation=nn.GELU(),
            batch_size=1, dim_param={}, param_names=tuple(), lat_param={}, **kwargs):
        super().__init__(dim_param, param_names, lat_param)
        self.mlp = CellMLP(layers=layers, in_dim=in_dim, out_dim=out_dim,
            inter_dims=inter_dims, activation= activation, batch_size=batch_size)
        self.batch_size = batch_size

    def forward(self, values, **kwargs):
        latent = self.mlp(values)
        parameters = self.lat2in(latent, (self.batch_size,))
        return self.in2term(parameters)


class Hist(nn.Module):
    def __init__(self, hist_len, extra_hist, seq_len, lat, device, 
                use_time_encoding = False, time_constant = 1.0, subdivisions=None):
        super().__init__()
        if subdivisions is None:
            subdivisions = [0.75]
        self.hist_len= hist_len
        self.seq_len = seq_len
        self.lat = lat
        self.device = device
        self.subdivisions = subdivisions
        self.use_time_encoding = use_time_encoding
        self.abs_sub_pos = []
        sub_rel = [0.0] + subdivisions + [1.0] #start, middle-subdivitions, end
        sub_itr = range(len(sub_rel)-1)
        if not use_time_encoding is None:
            #the time and the values are stored
            hist_time = torch.zeros(1, seq_len, 1, 1, device = device) #hist, seq, B(1), lat(1)
            now_t = time.time()
            for i in sub_itr:
                sub_poss = slice(
                    int(sub_rel[i]*seq_len), int(sub_rel[i+1]*seq_len)
                )
                self.abs_sub_pos.append(sub_poss)
                slice_len = sub_poss.stop - sub_poss.start
                for j in range(sub_poss.start, sub_poss.stop):
                    hist_time[0][j][0][0] = now_t + (((j-0.5)*(slice_len**i)) - 1.0)
                    #it makes past time more close to what it would expect, instead of a single value
                    #of time
            self.to_shift = ['hist', 'hist_time']
        else:
            self.to_shift = ['hist']
            hist_time = [None]*hist_len
            #with no time encoding it assumes the time encoding was already used before
        self.sub_itr = sub_itr
        self.sub_rel = sub_rel
        buffers = {
            'hist': torch.rand((hist_len + extra_hist, seq_len, 1, lat), device=device),
            'hist_time': hist_time.clone() * time_constant,
            'position': hist_time,
            'counter': torch.zeros(
                (hist_len, len(sub_rel)-1, 1, 1),dtype=torch.long, device=device
            ),
            #counter keeps track of the number of entries added to each
            # subdivision,
        }
        self.entries = {
            key:torch.zeros((hist_len, *buffers[key].size()[-2:]), device=device) \
                for key in self.to_shift
        }     
        self.buffers_keys = buffers.keys()
        for key, param in buffers.items():
            param.requires_grad = False
            self.register_buffer(key, param)

    def get_to_shift(self, keys):
        return [tsh for tsh in keys if tsh in self.to_shift]

    def average_with_entry(self, to_shift_values, index, n):
        to_shift = self.get_to_shift(to_shift_values.keys())
        for tsh in to_shift:
            values = self.entries[tsh][index]
            to_values = to_shift_values[tsh]
            self.entries[tsh][index] = dynamic_average_pyt(
                values, to_values, n)

    def next_cycle(self,):
        '''takes the entries of each cycle and average them to be added
            in the respective history, only the first hist index(encoder) is used for
            time because is the same for the decoder.
        '''
        buffers = dict(self.named_buffers(prefix='', recurse=False))
        for i in range(self.entries):
            if i == 0:
                to_shift = self.entries
            else:
                to_shift = {'hist':self.entries['hist']}
            self.shift_hist(to_shift, i)
        self.entries = {key:torch.zeros_like(val[:, 0]) for key,val in buffers.items()} 
            
    def shift_hist(self, to_shift_values, index = 0):
        ''' it takes the input tensor x and time and shifts it to the next
            position, if the next postion reach a position of next subdivision
            then the actual sudivision is averaged, and the counter is reseted,
            and this mean is placed in the next position of the next subdivision,
            if the next position of the next subdivision reaches yet another
            subdivision again this subdivision is averaged and placed in this 
            "next next subdivision"
            x: values, tensor
            time_now: float
            index: hist to use, 0 encoder, 1 decoder, int
        '''
        buffers = dict(self.named_buffers(prefix='', recurse=False))
        shift_sub = False
        to_shift = self.get_to_shift(to_shift_values.keys())
        for i in self.sub_itr:
            next_s = {}
            if i > 0:
                prev_ctr = buffers['counter'][index][i-1]
                prev_slc = self.abs_sub_pos[i-1]
                prev_seq_len = prev_slc.stop - prev_slc.start
                if (prev_ctr - 1) == prev_seq_len: #length of lower subdivision
                    slc = self.abs_sub_pos[i-1]
                    for tsh in to_shift:
                        sliced_v = buffers[tsh][index][slc]
                        next_s[tsh] = torch.mean(sliced_v, dim=0) #B, lat
                        #buffers[tsh][index][slc_n.start] = next_s
                    buffers['counter'][index][i-1] = 0 #reset prev counter
                    shift_sub = True
                else:
                    shift_sub = False
            else:
                next_s = to_shift_values
            if i == 0 or shift_sub:
                slc = self.abs_sub_pos[i]
                for tsh in to_shift:
                    buffers[tsh][index].requires_grad = True
                    shift = buffers[tsh][index][slc.start:slc.stop-1]
                    buffers[tsh][index][slc.start+1:slc.stop] = shift.clone()
                    buffers[tsh][index][slc.start] = next_s[tsh]
                if i < len(self.sub_itr) - 1:
                    buffers['counter'][index][i] += 1
        #for tsh in to_shift:
        #    buffers[tsh][index].requires_grad = True
        #    buffers[tsh][index][0] = to_shift_values[tsh]
        seq = buffers['hist'][index]
        buffers['hist'][index] = buffers['hist'][index].detach()
        for n, buff in buffers.items():
            buff.requires_grad = False
            self.register_buffer(n, buff)
        return seq
        #the hist sequence with the input value to backpropagate,
        #only used when the entries are not averaged.
class HistTransformer(CellModelBase):
    def __init__(self, seq_len, lat,
            device, subdivisions:list = None, use_neuro_mod:bool = True,
            use_hist:list = 'all', extra_hist:int = 0,
            use_time_encoding:str = 'all', time_constant:float = 1.0,
            mask_decoder_attn:bool = False, use_relative_modulator:bool = False,
            singular_modulation:bool = False,
            **kwargs:dict):
        '''
            Experimental transformer that uses as sequence de past inputs and
            the past outputs, also averages the inputs so the history doesn't
            get populated by recent inputs but instead by each cycle
            seq_len: length of past inputs, with some compressed
            lat: dimension of the latent space
            use_hist: use the encoder, decorder or both histories
            extra_hist: add an extra history mainly handled outside this class
            subdivisions: divides the sequence in different levels of averaging
                past inputs
            use_neuro_mod: flag that introduces a mlp network to the reduction
                in dimension of the output sequence to latent dimension
                otherwise it uses an average
            use_time_encoding: str if positional(or temporal) encoding has to
                be used either in the encoder, decoder or both
            time_constant: initial constant used to partition time as units of
                positions in the positional encoding
            mask_decoder_attn: if use a mask for the decoder so the transformer
                attends to future positions
            neural_saturation: reduces the signal of the sequences with standarization
                (kind of), with the mean and the standard deviation
            **kwargs: keyword arguments for pytorch's tranformer module class

        '''
        super().__init__()
        if subdivisions is None:
            subdivisions = [0.75]
        self.out_dim = lat
        self.device = device
        self.use_neuro_mod = use_neuro_mod
        self.lat = lat
        self.seq_len = seq_len
        self.use_hist = use_hist
        self.use_time_encoding = use_time_encoding
        if use_hist == 'all':
            hist_len = 2
        elif use_hist == 'enc':
            hist_len = 1
        else:
            raise ValueError('Unrecognized option for historical transformer ' + use_hist)
        self.hist_len = hist_len
        self.extra_hist = extra_hist
        self.hist = Hist(hist_len, extra_hist, seq_len, lat, device, 
            use_time_encoding=use_time_encoding, time_constant=1.0, 
            subdivisions=subdivisions)
        if mask_decoder_attn:
            #triangular mask used for the decoder
            mask = torch.triu(
                torch.ones(seq_len, seq_len,
                    device=device)*float('-inf'), diagonal=1
            )
        else:
            mask = None
        self.mask = mask
        
        if self.use_neuro_mod:
            self.neuro_mod_weight = MLP(layers = 3, in_dim=lat, out_dim=seq_len)
            self.neuro_mod_bias = MLP(layers = 3, in_dim=lat, out_dim=lat)
            self.neuro_mod_op = lambda x, w, b: torch.mv(
                torch.permute(x,(1,0,2)),w) + b #S, B, Lat -> B, S, Lat, then matrix vector mult
        else:
            self.neuro_mod_weight = lambda x: None
            self.neuro_mod_bias = lambda x: None
            self.neuro_mod_op = lambda x, w, b: x[-1]
            #reduce the sequential dimension
        if use_relative_modulator:
            self.relative_modulator = RelativeModulator(
                lat, singular_modulation)
        else:
            self.relative_modulator = lambda x: x

        if use_time_encoding == 'all':
            self.time_enc_fn_enc = self.time_encoding
            self.time_enc_fn_dec = self.time_encoding
        elif use_time_encoding == 'enc':
            self.time_enc_fn_enc = self.time_encoding
            self.time_enc_fn_dec = lambda x, y, z: x
        elif use_time_encoding == None:
            self.time_enc_fn_enc = lambda x, y, z: x
            self.time_enc_fn_dec = lambda x, y, z: x
        self.transformer = nn.Transformer(
            d_model=lat, **kwargs)
    
    def time_encoding(self, x, time_now, time_hist):
        ''' positional encoding using relative time instead of postion
            in the sequence
            x: tensor, Seq, B, Lat
            time_now: float, actual time
            time_hist: tensor, Seq, B, Lat, historical time
        '''
        rel_time = time_hist - time_now
        div_term = torch.exp(torch.arange(
            0, self.lat, 2, device=self.device) * (-math.log(10000.0) / self.lat))
        te = torch.zeros(self.seq_len, 1, self.lat, device=self.device) #Seq, B, lat
        te[:, :, 0::2] = torch.sin(rel_time * div_term)
        te[:, :, 1::2] = torch.cos(rel_time * div_term)
        return x + te

    def forward_shift(self, to_shift_values, i = 0):
        to_shift = self.hist.get_to_shift(to_shift_values.keys())
        buffers = dict(self.hist.named_buffers(prefix='', recurse=False))
        shifted = {key:None for key in to_shift}
        for key in to_shift:
            seq = buffers[key][i].clone() #seq, 1, lat
            shift = seq[1:].clone()
            seq[-1, :] = 0
            seq[:-1] = shift
            seq[-1] = to_shift_values[key]
            shifted[key] = seq 
        return shifted

    def forward_enc(self, x, time_now, n):
        '''forward function of the encoding sequence
            x: tensor: B, Lat
            time_now: tensor: B, 1
            n: times forwarded in cycle
        '''
        to_shift_values = {'hist': x, 'hist_time': time_now}
        self.hist.average_with_entry(to_shift_values, 0, n)
        shifted = self.forward_shift(to_shift_values)

        src_seq = self.time_enc_fn_enc(
            shifted['hist'], time_now, shifted['hist_time']
        )
        return src_seq, shifted['hist_time']

    def forward_dec(self, time_now, time_hist = None):
        '''forward function of the decoding sequence
            x: tensor: B, Lat
            time_now: float
        '''
        buffers = dict(self.hist.named_buffers(prefix='', recurse=False))
        tgt_seq = buffers['hist'][1]
        if time_hist is None:
            time_hist = buffers['hist_time'][0]
        tgt_seq = self.time_enc_fn_dec(tgt_seq, time_now, time_hist)
        return tgt_seq

    def forward_post_dec(self, out, n):
        '''forward function of the decoding sequence, after transformer
            out: tensor: B, Lat
            time_now: float
        '''
        self.hist.average_with_entry({'hist':out}, 1, n)

    def forward(self, x, time_now, n):
        src_seq, time_hist = self.forward_enc(x, time_now, n)
        tgt_seq = self.forward_dec(time_now, time_hist)
        
        out = self.transformer(  #S, B(1), Lat
            src_seq, tgt_seq, 
            tgt_mask=self.mask
        )
        out_1 = out[-1] #B, lat
        self.forward_post_dec(out_1, n)
        
        return out_1

class HeadMemHistTransformer(HistTransformer):
    ''' Historical transformer without decoder sequence processing
    '''
    def forward(self, x, time_now, n, tgt_seq):
        ''' x: B, lat
            time_now: float
            n: int
            tgt_seq = Seq, B, lat
        '''
        src_seq, _ = self.forward_enc(x, time_now, n)
        out = self.transformer(
            src_seq, tgt_seq, 
            tgt_mask=self.mask
        )
        return out

class MemHistTransformer(CellModelBase):
    def __init__(self, mem_len, lat, merge_we = False,
            lambda_mem_momentum = 0.001, reduction = 'normsum',
            device= None, **kwargs):
        ''' seq_len: int: sequence length
            lat: int: latent dimension
            merge_we: bool: to merge write operation with erase operation
                using a Tanh activation function, >0 writes, <0 erases
            lambda_mem_momemtum: float: momemtum of memory to not be
                modified, if a part of the memory is constantly read then
                is preserved, otherwise it can be more easily modified.
            device: torch.device
            **kwargs: dict: keyword arguments for the Hist transformer
        '''
        super().__init__()
        self.out_dim = lat
        seq_len = mem_len
        self.seq_len = seq_len
        self.lat = lat
        self.merge_we = merge_we
        self.lmm = lambda_mem_momentum
        self.device = device
        self.write_transformer = HistTransformer(seq_len, lat, 
            use_hist = 'enc', use_neuro_mod=False, extra_hist = 0, 
            use_time_encoding= 'enc', mask_decoder_attn= False,
            device=device, **kwargs)
        self.read_transformer = HeadMemHistTransformer(seq_len, lat, 
        use_hist = 'enc', use_neuro_mod=False, extra_hist = 0, mask_decoder_attn= False,
        use_time_encoding= 'enc', device=device,
        **kwargs)
        if not merge_we: #no merge of writing and erasing
            self.erase_transformer = HeadMemHistTransformer(seq_len, lat,
            use_hist='enc', use_neuro_mod = False, extra_hist = 0, mask_decoder_attn= False,
            use_time_encoding= 'enc', device=device, **kwargs)
        else:
            self.erase_transformer = None
        if reduction == 'last':
            self.reduction_fn = lambda x: x[-1]
        elif reduction == 'mean':
            self.reduction_fn = lambda x: torch.mean(x, dim=0)
        elif reduction == 'normsum':
            self.lnorm = nn.LayerNorm([1, lat], elementwise_affine=False)
            self.reduction_fn = lambda x: self.lnorm(
                torch.sum(x, dim=0))
        
        self.register_buffer(
            'memory', torch.zeros(mem_len, 1, lat, device=device)
        )
        memory_momentum = torch.zeros(mem_len, 1, lat, device=device)
        self.register_buffer(
            'memory_momemtum', memory_momentum
        )
        self.softmax = nn.Softmax(dim=0)
        self.mem_mask_entry = torch.zeros(mem_len, 1, lat, device=device)
        self.mem_entry = torch.zeros(mem_len, 1, lat, device=device)

    def next_cycle(self):
        ''' cleans the model of all previous values
        '''
        momemtum = self.get_buffer('memory_momemtum')
        nmom = self.mem_mask_entry
        self.register_buffer('memory_momemtum',
            torch.clamp(momemtum*(1-self.lmm) + nmom*self.lmm, 
            min = 0.0, max = 1.0)
        )
        self.register_buffer('memory', self.mem_entry)
        self.mem_mask_entry = torch.zeros_like(nmom)
        self.mem_entry = torch.zeros_like(self.mem_entry)
        self.write_transformer.next_cycle()
        self.read_transformer.next_cycle()
        if not self.erase_transformer is None:
            self.erase_transformer.next_cycle()

    def forward(self, x :torch.Tensor, time_now :float, maturity: float, n:int):
        '''x: B, L
            time_now: actual time
            maturity: maturity of the cell (not cycle)
            n: number of training cycles it has been forwarded
        '''

        memory = self.get_buffer('memory') #S,B,L
        momemtum = self.get_buffer('memory_momemtum') #S,B,L
        mmom = maturity*momemtum
        xr = x
        xr = self.read_transformer(xr, time_now, n, memory)  #S,B,L
        xrmask = torch.sigmoid(xr) #S,B,L
        invsum_xrmask = 1 - self.reduction_fn(xrmask) #B,L
        xr = (xrmask * memory) #S,B,L
        xr = self.reduction_fn(xr) #B,L
        out = xr + invsum_xrmask*x #B,L

        xw, _ = self.write_transformer.forward_enc(x, time_now, n) #S, B, L
        if not self.merge_we:
            xe = self.erase_transformer(x, time_now, n, memory)
            xwmask = torch.sigmoid(xw) * (1-mmom)  #S, B, L
            xemask = torch.sigmoid(xe) * (1-mmom) #S, B, L
        else:
            xwmask = torch.tanh(xw) * (1-mmom) #S, B, L
            xemask = torch.relu(-xwmask) #S, B, L
            xwmask = torch.relu(xwmask) #S, B, L
        entry_memory = memory*(1-xemask) + x.unsqueeze(0)*xwmask #S, B, L

        self.mem_mask_entry = dynamic_average_pyt(
            self.mem_mask_entry, xrmask.detach().clone(), n)
        self.mem_entry = dynamic_average_pyt(
            self.mem_entry, entry_memory.detach().clone(), n)

        outwe = memory-entry_memory #S, B, L
        outwe = self.reduction_fn(outwe) #Amount of memory modified
        out = out + outwe
        return out