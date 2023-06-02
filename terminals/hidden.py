import torch
import time
from terminals.generic import Terminal, InputRealTerminal
from generators.extra_models import losses
from generators import models
from utils import time_encoding, get_diffuseness_io_values

class Test(Terminal):
    ''' Test terminal used to test the internal methods
        and interactions with other terminals without fully executing the later
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.lat_param = {'values':128, 'values_sub': 64}
        self.dim_param = {'values': 64, 'values_sub': 32}

        self.param_names.append('values')
        self.param_names.append('values_sub')
        def tmp(name):
            self.set_parameter_settings(name, 
                dimensions=(self.batch_size, self.dim_param[name]),
                model=models.CellModelWithParameters,
            )
        self._param_function(['values', 'values_sub'], tmp)
        self.modify_parameter_settings('values', 
            model_settings={
                'lat': 256, 
                'lat_param': self.lat_param, 
                'dim_param': self.dim_param,
                'param_names': self.param_names,
                'batch_size': self.batch_size
            },
        )
        self.modify_parameter_settings('values_sub', parent='values')
        
class TestInput(Test):
    def __init__(self, settings):
        super().__init__(settings)
        self.modify_parameter_settings('values', 
            method_fn=self.random_input,
            model = models.InputParamMLP,
            model_settings = {
                'in_dim': sum(self.dim_param.values()),
                'out_dim': sum(self.lat_param.values()),
            }
        )

    def random_input(self):
        dim = self.parameters_settings['values']['dimensions']
        return torch.rand(dim, device = self.device)

class TestOutput(Test):
    def __init__(self, settings):
        super().__init__(settings)
        self.modify_parameter_settings('values', 
            loss_fn = losses.mse_loss_fn,
            model = models.OutputParamMLP,
            model_settings= {
                'in_dim': sum(self.lat_param.values()),
                'out_dim': sum(self.dim_param.values())
            }
        )
        
class InternalTerminal(Terminal):
    '''Terminal that is included as a partial residue in all
        the cell values
    '''
    def __init__(self, settings):
        super().__init__(settings)
        capacity = settings[
            'terminal_sockets']['InternalTerminal']['capacity']
        self.parameters_settings.update({
            'values':{
                'dimensions': (1,capacity),
                'tensortype': 'float',
                'values': None,
                'model': 'MLP',
                'latent_dim':[128],
                'model_settings': {
                    'layers':2,
                },
            },    
        })
        self.param_names.append('values')

class MemoryBank(Terminal):
    '''Terminal that keeps a state in form of memory
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings.update({
            'values':{
                'dimensions': (1, 256),
                'tensortype': 'float',
                'values': None,
                'model': 'CellMLP',
                'latent_dim':[256],
                'entropy':True,
                'model_settings': {
                    'layers':2,
                },
            },
        })
        self.param_names.append('values')

class SourceAttention(Terminal):
    '''Terminal that multiplies the input values of the cycle'''
    def __init__(self, settings):
        super().__init__(settings) 

class SystemSensors(Terminal):
    def __init__(self, settings):
        super().__init__(settings)
        self.batch_size = 1
        self.tick_components = self.settings[
            'internal_values_components']['tick_encoding']
        self.time_components = self.settings[
            'internal_values_components']['time_encoding']
        self.parameters_settings.update({
            'cycletime':{
                'dimensions': (self.batch_size, 1),
                'tensortype': 'float',
                'values': None,
                'model':'CellMLP',
                'latent_dim': 32,
                'entropy': False,
                'sequential_dim': None,
                'model_settings':{}
            },
            'daytime':{
                'dimensions': (self.batch_size, len(self.time_components)*2),
                'tensortype': 'float',
                'values': None,
                'model':'CellMLP',
                'latent_dim': 32,
                'entropy': False,
                'sequential_dim': None,
                'model_settings':{}
            },
            'routings':{
                'dimensions': (self.batch_size, 1),
                'tensortype': 'float',
                'values': None,
                'model':'CellMLP',
                'latent_dim': 32,
                'entropy': False,
                'sequential_dim': None,
                'model_settings':{}
            },
            'ticking':{
                'dimensions': (self.batch_size, len(self.tick_components)),
                'tensortype': 'float',
                'values': None,
                'model':'CellMLP',
                'latent_dim': 32,
                'entropy': False,
                'sequential_dim': None,
                'model_settings':{}
            },
            'losses':{
                'dimensions': (self.batch_size, None),
                'tensortype': 'float',
                'values': None,
                'model':'CellMLP',
                'latent_dim': 32,
                'entropy': False,
                'sequential_dim': None,
                'model_settings':{}
            }
        })
        self.param_names += self.settings[
            'terminal_sockets'](self.__str__)['sensors']

class SystemSensorsOutputEncoder(SystemSensors):
    def __init__(self, settings):
        super().__init__(settings)
        self.parameters_settings['cycletime']['method_fn'] = self.get_cycletime
        self.parameters_settings['daytime']['method_fn'] = self.get_daytime
        self.parameters_settings['routings']['method_fn'] = self.get_routings
        self.parameters_settings['tickings']['method_fn'] = self.get_tickings
        self.parameters_settings['losses']['method_fn'] = self.get_losses

    def get_cycletime(self, Hmemory, phase):
        cycle_time = torch.tensor(Hmemory.last_cycle_time, device=self.device)
        return torch.stack([cycle_time]*self.batch_size)

    def get_daytime(self, Hmemory, phase):
        now_tt = time.time()
        encoding = torch.tensor(
            time_encoding(now_tt, self.time_components, flatten=True),
            device=self.device
        )
        return torch.stack([encoding]*self.batch_size)

    def get_routings(self, Hmemory, phase):
        routings_frac = Hmemory.last_routings/self.settings[
            'cell_limits']['routing']['max']
        routings_frac = torch.tensor(routings_frac, device=self.device)
        return torch.stack([routings_frac]*self.batch_size)

    def get_tickings(self, Hmemory, phase):
        tick_tensor = torch.tensor(Hmemory.tickings, device=self.device)
        return torch.stack([tick_tensor]*self.batch_size)

    def get_losses(self, Hmemory, phase):
        losses_tensor = torch.tensor(Hmemory.last_losses, device=self.device)
        return torch.stack([losses_tensor]*self.batch_size)

SystemSensorsOutputCritic = SystemSensorsOutputEncoder

class KillTerminal(Terminal):
    '''Special terminal that is traversed once a cell dies, when the
        cell doesn't take any action
    '''
    def __init__(self, settings):
        super().__init__(settings)
        self.param_names = []
        self.parameters_settings = {}

    def set_diffuseness_io_values(self):
        '''sets the diffusenes values in diff_values'''
        settings = self.settings
        diff_values = get_diffuseness_io_values(
            settings['diffuseness_dim'], self.device,
            False)
        diff_values.requires_grad = False

        self.diff_values = diff_values

    def clean_next(self,):
        '''cleans terminal for next cycle'''
        pass
        #self.chains = []
        #self.parameters_to_back = []
        #self.graph_diff = None
        #self.graph_diffs = []

    def set_cell(self, *args, **kwargs):
        pass
    
    def restore_parameters(self, *args, **kwargs):
        pass