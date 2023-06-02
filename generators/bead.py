import torch
import torch.nn.functional as F
import os
import pathlib
import shutil
import time
from generators.cellmodels import TerminalCellModel
from generators.cell import Cell

class Bead(Cell): 
    '''Terminal analoge'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class BeadMemory(Bead):
    '''Changes the settings so it uses memory
    '''
    def __init__(self, **kwargs):
        if 'cell_model_settings' in kwargs:
            kwargs['cell_model_settings'] = kwargs['cell_model_settings'].copy()
        else:
            kwargs['cell_model_settings'] = {}
        kwargs['cell_model_settings']['use_memory'] = True
        super().__init__(**kwargs)

class TerminalBead(Cell):
    '''Cell object for terminals, has unique declaration model method
    '''
    def __init__(self, terminal, CellModelClass):
        self.terminal = terminal
        super().__init__(
            CellModelClass = CellModelClass,
            **terminal.cell_arguments
        )
        self.type = 'terminal'
        self.io_index = self.settings['terminal_io'].index(
            self.terminal.category_dict['io'])

    def declare_model(self,):
        '''declares the model with terminal and model class for that terminal
        '''
        return self.CellModelClass(
            terminal = self.terminal,
            ModelClass = self.MainModelClass, model_settings = self.model_settings, 
            settings = self.settings, device= self.device,
            **self.cell_model_settings
            ).to(self.device)

    def get_cell_count_fraction(self,):
        '''processes cell counts'''
        termcell_counts = self.counts #cells
        termcell_f = F.normalize(termcell_counts, dim=0)
        return termcell_f

class TerminalBeadInput(TerminalBead):
    def __init__(self, terminal):
        super().__init__(terminal, TerminalCellModel)
    
    def update_diffuseness(self, fractions, stats, n):
        '''keeps input diffuseness static'''
        fractions = torch.stack([stats['diffuseness'][0], fractions[1]])
        super().update_diffuseness(fractions, stats, n)

class TerminalBeadOutput(TerminalBead):
    '''Because the output terminal doesn't route to anything
        there's no need for subactions, even memory
    '''
    def __init__(self, terminal):
        terminal.cell_arguments[
            'cell_model_settings']['use_subactions'] = []
        super().__init__(terminal, TerminalCellModel)
        self.diff_index += self.settings['diffuseness_dim']

    def create_statistics(self, ):
        stats = super().create_statistics()
        stats['reward_trained'] = 0.0
        return stats
    
    def get_cell_count_fraction(self):
        cell_f = super().get_cell_count_fraction()
        self.model.update_activation_hist(cell_f)
        return cell_f

    def update_diffuseness(self, fractions, stats, n):
        '''keeps output diffuseness static'''
        fractions = torch.stack([fractions[0], stats['diffuseness'][1]])
        super().update_diffuseness(fractions, stats, n)

class OffCellModel(object):
    '''Dummy cell model used for saving the model in CPU, so it
        can use the same methods as the cells in GPU for saving and
        loading
    '''
    def __init__(self,):
        self.memory_slots = None
        self.residue = None
        self.model_state = {}
    def state_dict(self):
        return self.model_state
    def load_state_dict(self, state):
        self.model_state = state

class OffOptimizer(object):
    '''Dummy optimizer for saving the parameters, used in OffCellModel
        objects
    '''
    def __init__(self,):
        self.opt_state = {}
    def state_dict(self):
        return self.opt_state
    def load_state_dict(self, state):
        self.opt_state = state

class OffCell(Cell):
    ''' Dummy cell for saving and loading the checkpoint, non functional
    
    '''
    def __init__(self, name, mainfolder, settings):
        self.name = name
        self.statistics = {}
        self.optimizer = OffOptimizer()
        self.model = OffCellModel()
        self.device = torch.device('cpu')

        self.mainfolder = mainfolder
        self.folder_path = os.path.join(mainfolder, name)
        self.settings = settings
        self.chk_path = os.path.join(
            self.folder_path, settings['memory']['checkpoint_path']
        )

    def offload_cpu(self):
        self.save_state()
        self.__init__(self.name, self.mainfolder, self.settings)