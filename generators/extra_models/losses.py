import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss_fn(val0:torch.Tensor, val1:torch.Tensor):
    ''' val0: tensor to backprop, any dim
        val1: tensor
    '''
    return F.mse_loss(val0, val1)

def categorical_loss_wrp(seq_dim:int):
    ''' val0: tensor to backprop, logits
        val1: tensor, target logits
    '''
    def tmp(val0:torch.Tensor, val1:torch.Tensor):
        batch_dim = 1 if seq_dim == 0 else 0
        val0 = val0.reshape(val0.size()[batch_dim], -1)
        val1 = val1.reshape_as(val0)
        val1_int = torch.argmax(val1, dim=0)
        loss = F.cross_entropy(val0, val1_int)
        return loss
    return tmp

def glll_loss_fn(val0:torch.Tensor, val1:torch.Tensor):
    pass
    
