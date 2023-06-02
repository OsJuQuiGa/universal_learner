from generators.extra_models.dnc.sdnc import SDNC
import torch
import torch.nn.functional as F
from utils import dynamic_average_pyt

class ModSDNC(SDNC):
    def __init__(self, lat, batch_size, device, **kwargs):
        dtype = device.type
        if dtype == 'cuda':
            if device.index is None:
                gpu_id = 0
            else:
                gpu_id = device.index
        else:
            gpu_id = -1
        super().__init__(lat, lat*2, gpu_id=gpu_id, batch_first=False, **kwargs)
        self.device = device
        self.batch_size = batch_size
        hx = self._init_hidden((None, None, None), batch_size, False)
        self.hx = hx

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['_hx'] = self.hx

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.hx = state_dict['_hx']
        return super().load_state_dict(state_dict, strict)

    def forward(self, values, reset_experience=False, pass_through_memory=True):
        hx = self.hx
        values, (hx2) = super().forward(values, hx, 
            reset_experience, pass_through_memory)
        self.hx = hx2
        return values


             

