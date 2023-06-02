import os
import time
import math
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import generators.extra_models.losses as loss_fns
import functools
import random
import copy
from utils import (
    get_normal_loss_sample, get_category, check_key,
    check_make_folder, get_diffuseness_io_values,
    get_routing_masks, param_function_return, param_function,
    update_dict
)
from generators.bead import TerminalBeadInput, TerminalBeadOutput
import generators.models as Models
#from terminals.forward import _forward_dict_fn
#class Parameter(object):
def forward_wrap(func):
    def inner(*args, **kwargs):
        obj = args[0]
        obj.forward_ret = obj.forward_fn(*args, **kwargs)
        return obj.forward_ret
    return inner

class Terminal():
    '''
        Terminal constructor
        settings: dict:
        initializes all the variables needed for operation
    '''
    #StaticVariables
    slave = True
    index = None
    full_index = None
    number = None
    mainfolder = None
    term_settings = {
        'batch_size': 1,
        'imitate_master': False,
    }
    category_dict = {}
    device = torch.device('cpu')

    cell_settings = None
    main_model = None

    # io = {# source:Screen, Pointer, Sound
    # type = Real, Virtual, Encoder, Dream, Critic
    # io = Either input, output or hidden, 
    # name = extra name}
    @classmethod
    def get_static(cls):
        return cls.category_dict, cls.term_settings, cls.cell_settings, cls.main_model
   
    def __init__(self, settings):
        #mainfolder, folder where terminal's values are stored
        self.settings = settings
        (self.category_dict, self.term_settings, self.main_model, 
            self.cell_settings) = self.get_static()
        self.batch_size = self.term_settings['batch_size']
        self.imitate_master = self.term_settings['imitate_master'] and \
            self.settings['batch_imitation'] and \
            self.slave
    
        self.batch_size *= 2 if self.imitate_master else 1
        self.base_counter_subcat = {}
        self.settings = settings
        self.category = get_category(self.category_dict)
        self.folder_path = os.path.join(self.mainfolder, self.category)
        self.create_paths()
        self.parameters = {}
        self.param_names = []
        self.parameters_settings = {}
        self.extra_parameters = {} #past interpretation in the case of the encoderoutput
        #delayed input for input terminals
        self.parameters_settings = self.parameters_settings
        self.has_real_expression = False
        self.gather_settings = {}
        if self.main_model is None:
            self.term_settings['main_model'] = settings['main_model']
        if self.cell_settings is None:
            self.term_settings['cell_settings'] = {}
        model_settings = copy.deepcopy(
            self.settings['main_model_parameters'][self.term_settings['main_model']])
        model_settings.update(self.term_settings['cell_settings'])
        self.cell_arguments = {
            'name': self.category,
            'mainfolder': self.mainfolder,
            'settings': self.settings,
            #'cell_settings': {
            #    'MainModelClass': getattr(Models, self.term_settings['main_model'])
            #},
            'MainModelClass': getattr(Models, self.term_settings['main_model']),
            'model_settings': model_settings
        }
        self.subaction_settings = self.settings['terminal_subaction'].copy()
        
        self.cell = None
        self.diff_values = None
        self.init_time = 0.0
        self.cycle_time = 0.0
        self.shared_category_dict = self.category_dict
        self.shared_category = self.category
        self.shared_intervalues = None
        self.r_masks = None #routing priority masks, how much to score each terminal

    def set_cell(self, init_behaviour = 'set_and_load', terminals = None,
            phase = None):
        '''sets the cell of the terminal with the mode (init_behaviour)
            also the diffuseness values of the terminal
            init_behaviour: str
            ----return cell_obj
        '''
        self.cell_arguments['device'] = self.device #The device is set by HierarchicalMemory object
        self.cell_arguments['init_behaviour'] = init_behaviour
        self.cell_arguments['terminals'] = terminals
        self.cell_arguments['diff_index'] = self.settings[
            'n_cells'] + self.index
        self.cell_arguments['terminals'] = terminals
        self.diff_values = get_diffuseness_io_values(
            self.settings['diffuseness_dim'],
            self.device, random=True
        )
        if self.category_dict['io'] == 'Output':
            self.cell = TerminalBeadOutput(self)
        elif self.category_dict['io'] == 'Input':
            self.cell = TerminalBeadInput(self)
        typ = self.category_dict['type'] 
        if typ in self.settings['phase_terminal_types_to_learn'][phase['wph']]:
            self.cell.train = bool(self.cell.train * True)
        else:
            self.cell.train = False
        return self.cell

    def set_parameter_settings(self, name:str, dimensions=(None, None),
            latent_dim=32, values = None, model= None, model_settings = None,
            sequential_dim=-1, categorical= False, tensortype='float', 
            loss_fn= None, method_fn = None,
            imitate = False, parent=None):
        self.parameters_settings[name] = {
            'dimensions':dimensions,
            'tensortype': tensortype,
            'values': values,
            'model': model,
            'latent_dim': latent_dim,
            'model_settings': {} if model_settings is None else model_settings,
            'sequential_dim': sequential_dim,
            'categorical': categorical,
            'loss_fn': loss_fn,
            'method_fn': method_fn,
            'imitate': imitate,
            'parent': parent,
        }
    def modify_parameter_settings(self, name:str, **kwargs):
        update_dict(self.parameters_settings[name], kwargs)

    def create_parameters(self, extra = ''):
        '''Creates the parameteres defined in param_values
            extra: str: extra set of parameters
        '''
        param_dict = self._param_to_use(extra)
        def tmp(name):
            values = self._create_parameter(name)
            param_dict[name] = values
        self._param_function(self.parameters, tmp)

    def _create_parameter(self, name):
        '''Creates a Parameter for input output, Real Virtual, etc...
         DEF:Parameter= Variable that holds a tensor value of real-world or virtual(made by the NN)
        observations, it has specific dimensions, dtype
        name: str: name of the parameter
        device: torch.device
        '''
        sett = self.parameters_settings[name]
        dft_sett = self.settings['default_parameter'].copy()
        dft_sett.update(sett)
        sett = dft_sett
        if sett['values'] is None:
            values = torch.rand(*sett['dimensions'], device = self.device)*0.1 
        else:
            values = sett['values']
            if torch.is_tensor(sett['values']):
                values.to(device = self.device)
            else:
                values = torch.tensor(values, device = self.device)
        return values


    def create_paths(self, model_path = 'model'):
        '''Creates the path of the model and statistics
            model_path:str: subpath of the model
        '''
        check_make_folder(self.folder_path)
        self.parameters_folder = os.path.join(self.folder_path, '_parameters')
        check_make_folder(self.parameters_folder)

    def get_counter_category(self, counter_subcat = {}):
        '''Returns the same items in self.io but with the ones specified by counter_cat_dict
            counter_subcat: dict: keys, subcategories to change, values the subcategory
            ----return (counter_category:str, newcat_dict:dict)
        '''
        new_cat_dict = self.category_dict.copy()
        new_cat_dict.update(counter_subcat)
        counter_category = get_category(new_cat_dict)
        return counter_category, new_cat_dict

    def get_counter_folder(self, counter_subcat = {}, extra = None):
        '''gets folder of the counter_category
            counter_subcat: dict: keys, subcategories to change, values the subcategory
            ----return folder_path:str
        '''
        category,_ = self.get_counter_category(counter_subcat)
        folder_path = os.path.join(self.mainfolder, category)
        if extra:
            folder_path = os.path.join(folder_path, '_' + extra)
        return folder_path

    def restore_parameters(self):
        '''loads and sets the parameters from disk'''
        parameters = self.load_parameter_values()
        self.set_parameter_values(parameters)

    def load_parameter_values(self, counter_subcat = {}, extra = None, one_file=False):
        '''loads the parameters from disk
            counter_subcat:dict
            extra:str
            one_file: bool: it stores all parameters in one file instead of file
            per parameter
            ----return parameters: dict, {key: parameter name, value:tensor}
        '''
        folder_path = self.get_counter_folder(counter_subcat, extra)
        # if there's no values then it uses the initialized values when a parameter is created
        values2return = {}
        if os.path.exists(folder_path):
            if one_file:        
                values2return = torch.load(
                    os.path.join(folder_path, '_parameters.pth'),
                    map_location=self.device
                )
            else:
                def tmp(name):
                    mpath = os.path.join(folder_path, name + '.pth')
                    if os.path.exists(mpath):
                        loaded = torch.load(
                            mpath,
                            map_location=self.device
                        )
                    else:
                        loaded = None
                    return loaded
                values2return = self._param_function_return(
                    self.param_names, tmp)
        else:
            values2return = self.get_parameter_values()
        return values2return

    def save_parameter_values(self, counter_subcat = {}, extra = None, one_file=False):
        '''Saves paramters to disk, see terminal.load_parameter_values for arguments
            ----return values2save: dict
        '''
        folder_path = self.get_counter_folder(counter_subcat, extra)
        values2save = self.get_parameter_values(extra = extra, one_file=one_file)
        check_make_folder(folder_path)
        if one_file:        
            torch.save(
                values2save, os.path.join(folder_path, '_parameters.pth')
            )
        else:
            def tmp(name):
                assert not name == 'checkpoint'
                return torch.save(
                    values2save[name], os.path.join(folder_path, name + '.pth')
                )
            self._param_function([], tmp)
        return values2save

    def terminal_files_deletion(self,):
        '''Removes the folder that contains the values'''
        shutil.rmtree(self.folder_path)

    def _param_function_return(self, param_names:list = None, p_func = lambda x:(x)):
        '''Aux wrapper function for methods related to parameters
            param_names:list
            p_func:func
            ----return r_v:dict
        '''
        return param_function_return(self, param_names, p_func)

    def _param_function(self, param_names:list = None, p_func = lambda x:(x)):
        '''The same as _param_function_return without return'''
        param_function(self, param_names, p_func)

    def _check_values(self, param, values):
        '''checks the values tensor if it has the correct dimensions 
        and type for the parameter
        param = parameter
        values = tensor values'''
        assert tuple(values.size()) == param['dimensions']
        assert values.dtype == param['tensortype']
        
    def _param_to_use(self, extra = None):
        '''Chooses set of parameters to use given extra, 
            extra:str
            ----return param_dict: dict: same as parameters'''
        if extra is None or extra == '':
            param_dict = self.parameters
        else:
            if not extra in self.extra_parameters:
                self.extra_parameters[extra] = {}
            param_dict = self.extra_parameters[extra]
        return param_dict

    def set_parameter_values(self, values_dict, extra = None, check = False, 
            detach_copy = False, no_grad = False):
        '''sets the values given as values of the parameters of the terminal
        values_dict: dict: same shape as parameters
        extra: str: extra parameters to look for
        check: bool: check type of tensor to make sure the correct type is set
        '''
        param_dict = self._param_to_use(extra)
        def tmp(name):
            if name in self.param_names:
                if values_dict[name] == None:
                    print('Parameter ' + name + ' not loaded: ' + self.category)
                else:
                    if check:
                        self._check_values(
                            self.parameters_settings[name], values_dict[name])
                    if detach_copy:
                        values_dict[name] = values_dict[name].detach().copy()
                    if no_grad:
                        values_dict[name].requires_grad = False
                    param_dict[name] = values_dict[name]
        self._param_function(values_dict.keys(), p_func=tmp)

    def get_parameter_values(self, param_names = None, extra = None, 
            as_list = False, one_file=False, clone = False):
        '''retrieves the tensor values of parameters of the key in get_from
        param_names: list: mask of parameters to loop
        extra: str: set of parameters to look for, none is the default one
        as_list: bool: return parameters in list form
        one_file: bool:
        clone: bool: it detaches and clones the tensor that are gotten
        '''
        param_dict = self._param_to_use(extra)
        if one_file:
            param_values = param_dict
        else:
            def tmp(name):
                if clone:
                    return param_dict[name].detach().clone()
                else:
                    return param_dict[name]
            param_values = self._param_function_return(param_names, tmp)
            if as_list:
                param_values = [self.parameters[x] for x in self.param_names]
        return param_values

    def set_diffuseness_io_values(self):
        ''' sets the diffuseness values with the keys in settings
        '''
        #category_dict = self.category_dict
        settings = self.settings
        diff_values = torch.zeros(
            1, settings['diffuseness_dim'], device = self.device
        )
        diff_values.requires_grad = False
        diff_values[0, self.full_index] = 1.0
        self.diff_values = diff_values

    def forward_fn(self, Hmemory, phase):
        '''method done at init or end of cycle'''
        return self.parameters
    #def cycle_reset(self,):
    def express_values(self, parameters):
        ''' parameters
        '''
        pass
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#------------------------------INPUT OUTPUT TERMINALS ----------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

class OutputTerminal(Terminal):
    '''Output terminal constructor'''
    def __init__(self, settings):
        '''gather extra variables for output, adds reward to param_name
            settings:dict
        '''
        super().__init__(settings)
        _, self.counter_io_category_dict = self.get_counter_category(
            {'io':'Input'}
        )
        self.cell_cls_name = 'TerminalCellModelOutput'
        self.parameters_settings['reward'] = {
                'dimensions':(1,1),
                'tensortype': 'float',
                'values': None,
                'model': 'MLP',
                'latent_dim':64,
                'model_settings':{
                    'layers':4,
                },
                'loss_fn': F.mse_loss, #for real/virtual the cycle reward
                #for critic is the prediction
            }
        self.param_names.append('reward')
        self.traversed = False
        self.cycle_reward = 0

    def clean_next(self,):
        '''cleans the terminal for the next cycle, detaching parameters from back
            graph'''
        self.traversed = False
        for epn, val in self.extra_parameters.items():
            if isinstance(val, list):
                pass
            else:
                val.detach_()

    def equalize_dims(self, param_names:list = None, param2=None):
        ''' makes parameters have the same dimension'''
        param1 = self.parameters
        def tmp(name):
            seq_dim = self.parameters_settings[name]['sequential_dim']
            if not seq_dim is None:
                dim1 = param1[name].size()
                dim2 = param2[name].size()
                if dim1[seq_dim] != dim2[seq_dim]:
                    zdim = list(dim1)
                    zdim[seq_dim] = abs(dim1[seq_dim] - dim2[seq_dim])
                    zeros = torch.zeros(zdim, device=self.device)
                    if dim1[seq_dim] > dim2[seq_dim]:
                        param2[name] = torch.cat((param2[name], zeros))
                    else:
                        param1[name] = torch.cat((param1[name], zeros))
        self._param_function(param_names, p_func=tmp)
        
    def take_base_loss(self, param_names:list = None, param2 = None):
        ''' takes losses for each parameter, for Encoder and Critic
            and dream variant
            ---return parameters: dict
        '''
        param1 = self.parameters
        self.equalize_dims(param_names, param2)
        def tmp(name):
            loss_fn = self.parameters_settings[name]['loss_fn'] #must have
            return loss_fn(param1[name], param2[name])
        param_loss = self._param_function_return(
            param_names, p_func = tmp)
        return param_loss

    def take_reward_loss(self, param0, 
            param_closs = None, past_param_closs=None,
            l_cd = 1.0, l_var = 0.5):
        ''' takes losses for each parameter for reward, for Real and Virtual
            and dream variant
        '''
        cycle_reward = self.cell.cycle_reward
        param1 = self.parameters
        actions = self.extra_parameters['actions']
        if 'reward' in param1.keys():
            value = param1['reward']
            value_fn = self.parameters_settings['reward']['loss_fn']
            value_loss = value_fn(value, cycle_reward)
            delta = cycle_reward - value
            norwd_pn = self.param_names.copy().remove('reward')
        else:
            norwd_pn = self.param_names
            delta = 0

        param_dc_loss = self.get_diff_critic_loss(norwd_pn, 
            param0=param0, param_closs=param_closs,
            past_param_closs=past_param_closs)
        #it takes loss as regression of distributions

        def tmp(name):
            policy = param1[name]
            action = actions[name]
            loss_fn = self.parameters_settings[name]['loss_fn']
            #action could be float if gaussian, or int if categorical
            loss = loss_fn(policy, action) #the policy could be without action for categorical
            var_loss = 0
            if 'variance_loss_fn' in self.parameters_settings[name]:
                var_loss_fn = self.parameters_settings[name]['variance_loss_fn']
                var_loss = var_loss_fn(policy)
            crwd = l_cd*param_closs[name] + (1-l_cd)*param_dc_loss[name] + var_loss*l_var
            loss = -delta*loss + crwd*loss
            return loss
        param_rloss = self._param_function_return(norwd_pn, tmp)
        if 'reward' in param1.keys():
            param_rloss['reward'] = value_loss
        return param_rloss


    def get_diff_critic_loss(self, param_names, 
            param0, param_closs=None, past_param_closs=None):
        '''takes the approximate loss derived from the past loss
        and the actual loss of the critic terminal, and from past
        output
        param0: past parameters
        dcloss: difference in critic loss
        closs: critic loss
        '''
        self.equalize_dims(param_names, param0)
        param1 = self.parameters
        def tmp(name):
            p0n = param0[name] #past parameters
            p1n = param1[name] #actual parameters
            p1nd = p1n.detach().clone()
            k = p0n/(p1nd+1e-5)
            dclossh = torch.tanh(
                param_closs[name] - past_param_closs[name]
            )
            loss = F.mse_loss(p1n*k, -(p1nd-p0n)*dclossh + p0n)
            return loss
        param_loss = self._param_function_return(
                param_names, p_func=tmp)
        return param_loss
    
    def latent_to_parameter(self, values):
        ''' method for converting from latent space from cell to
            parameters
            values: tensor
        '''
        cm = self.cell.model
        parameters_ls = cm.forward_term(values)
        parameters = dict(zip(cm.pnb_used, parameters_ls))
        return parameters

    def expression_end_cycle(self, values):
        ''' Forwards the gathering or expressing of terminal's values
         end of chain
        values: tensor
        '''
        parameters = self.latent_to_parameter(values)
        self.parameters_to_back = parameters
        self.parameters = parameters
        self.express_values(parameters)

    def express_values(self, param):
        '''values that are exported to the world or other terminals'''
        #self.set_parameter_values(param)
        pass

    def lower_parameters_t(self, time_exp):
        '''lowers the parameters with time decay
            time_exp: float: time decay
        '''
        def tmp(name):
            self.parameters[
                name] = self.parameters[name] * time_exp
        self._param_function(p_func=tmp)

    def forward_fn(self, Hmemory, phase):
        '''generic forward for output, just store prev value'''
        self.set_parameter_values(
            self.parameters, 'prev_parameters', detach_copy= True,
            no_grad= True
        )
        return self.extra_parameters['prev_parameters']
        
class InputTerminal(Terminal): 
    '''Input Terminal constuctor'''
    def __init__(self, settings): 
        '''Inits generic parent terminal and gets counter cat_dict'''
        super().__init__(settings)
        _, self.counter_io_category_dict = self.get_counter_category(
            {'io':'Output'}
        )
        
    def generate_random_parameter_values(self, minmax=[-1, 1]):
        '''Generates random uniform values with the dimensions of the values
            minmax: Range of the random uniform numbers
            ---- return parameters:dict
        '''
        def tmp(name):
            rand = torch.rand(
                self.parameters_settings[name]['dimensions'],
                device=self.device
            )
            return rand * (minmax[1] - minmax[0]) + minmax[0]
        return self._param_function_return(p_func = tmp)

    def set_routing_masks(self):
        ''' Sets the routing mask
        '''
        pass
    def clean_next(self, ):
        ''' Cleans the shared values
        '''
        self.shared_intervalues = None

    def forward_fn_set(self, Hmemory, phase, strict = True, typ =None):
        '''for virtual mainly, sets the parameters to the counter io
            Hmemory: Hmemory_obj
            phase: dict
            strict: bool, use exactly the param names
            typ: str, counter type
        '''
        cat = self.category_dict
        source = cat['source']
        typ = typ if typ else cat['type']
        self.set_parameter_values(Hmemory.get_terminal_parameters(
            self.param_names, 'Output', (typ, source),
            self.device, strict = strict
        ))

    def forward_fn(self, Hmemory, phase):
        '''Returns in list form for expresing
        Hmemory: Hmemory_obj
        phase:dict
        ----return parameters:list
        '''
        return self.get_parameter_values(as_list=True)

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#------------------------------TYPE TERMINALS ----------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

################# OUTPUT TERMINAL SPECIFIC #############

OutputVirtualTerminal = OutputTerminal

class OutputRealTerminal(OutputTerminal):
    def __init__(self, settings):
        super().__init__(settings)
        self.batch_size = self.term_settings['batch_size']

OutputDreamTerminal = OutputTerminal

class OutputEncoderTerminal(OutputTerminal):
    '''creates old interpretations for dreaming phase, also extra
        sets of parameters for param_loss and param_difference of
        the past
        settings:dict
    '''
    def __init__(self, settings):
        super().__init__(settings)
        #self.cell_settings['model_settings']['in_dim'] *= 2
        self.extra_parameters['old_interpretations'] = []
        self.max_old_interpretations = 100
        self.last_loss = 0.050
        self.last_reward = 0 
        #def tmp(name):
        #    self._create_parameter(name, self.device)
        #self.last_param_loss = self._param_function_return(
        #    p_func=tmp)
        #self.last_param_difference = self._param_function_return(
        #    p_func=tmp)
        
    def restore_parameters(self):
        '''More work for the extra parameters
        '''
        super().restore_parameters()
        self.set_parameter_values(
            self.load_parameter_values(extra='interpretation'),
            extra='interpretation'
        )
        old_inter = self.load_parameter_values(
            extra = 'old_interpretations', one_file = True)
        if len(old_inter) > self.max_old_interpretations:
            old_inter = old_inter[:self.max_old_interpretations]
        self.extra_parameters['old_interpretations'] = old_inter

        self.loss_change = torch.tensor(
            self.loss_change, device=self.device)
        self.last_loss = torch.tensor(
            self.last_loss, device=self.device)

    def create_parameters(self):
        '''extra parameter interpretation'''
        super().create_parameters(extra)
        super().create_parameters(extra='interpretation')

    def save_parameter_values(self,):
        '''extra parameter interpretation and old interpretations'''
        super().save_parameter_values()
        super().save_parameter_values(extra='interpretation')
        super().save_parameter_values(
            extra='old_interpretations', one_file=True)

    def lower_parameters_t(self, time_exp):
        super().lower_parameters_t(time_exp)
        self.last_loss *= time_exp

    def forward_fn(self, Hmemory, phase):
        '''stores the past intepretation with p probability
            Hmemory: Hmemory_obj
            phase:dict
            ----return (
                parameters:dict
                sample:dict: sample from the past or from the real or
                    virtual terminals
            )
        '''
        if phase['wake']:
            typ = 'Real' if self.has_real_expression else 'Virtual'
        else:
            typ = 'Dream' if self.has_real_expression else 'VirtualDream'
        sample = Hmemory.get_terminal_parameters(
                self.param_names, 'Input',
                (typ, self.category_dict['source']), self.device,
                strict = False
            )
        if 'reward' in self.param_names:
            rwd = self.cell.cycle_reward
            self.parameters['reward'] = rwd

        if random.random() < 0.0001:
            old_int = self.extra_parameters['old_parameters']
            if len(old_int) == self.max_old_interpretations:
                idx = random.randint(1,self.max_old_interpretations)
                del(old_int[idx])
            old_int.append(self.get_parameter_values(clone=True))
        return self.parameters, sample

class OutputCriticTerminal(OutputEncoderTerminal):
    def forward_fn(self, Hmemory, phase):
        '''retrieves past interpretation and encoder forward
            Hmemory: Hmemory_obj
            phase:dict
            ----return (
                sample:dict: sample from the past or from the real or
                    virtual terminals,
                past_interpretation: dict: parameters of the last cycle
            )
        '''
        _, sample = super().forward_fn(
            Hmemory, phase
        )
        #past_interpretation = self.extra_parameters['interpretation']
        #self.set_parameter_values(
        #    self.get_parameter_values(clone=True)
        #)
        return sample

################# INPUT TERMINAL SPECIFIC #############

class InputRealTerminal(InputTerminal):
    def __init__(self, settings):
        super().__init__(settings)
        cat_dict = self.category_dict
        ctr_cat_dict = self.get_counter_category({'type':'Critic'})
        self.shared_category_dict = ctr_cat_dict
        self.shared_category = get_category(cat_dict)
        #input real uses critic's terminal values

    def set_routing_masks(self):
        r_masks = get_routing_masks(
            self.settings,
            cell_types ='all',
            types = ['Real', 'Dream'],
            sources = 'all',
            devices = self.device
        )
        self.r_masks = r_masks     
        
class InputVirtualTerminal(InputTerminal):
    def set_routing_masks(self):
        r_masks = get_routing_masks(
            self.settings,
            cell_types ='all',
            types = ['Real', 'Virtual'],
            sources = 'all',
            devices = self.device
        )
        self.r_masks = r_masks

class InputDreamTerminal(InputTerminal):
    def set_routing_masks(self):
        r_masks = get_routing_masks(
            self.settings,
            cell_types ='all',
            types = ['Dream'],
            sources = 'all',
            devices = self.device
        )
        self.r_masks = r_masks

    def forward_fn(self, Hmemory, phase):
        '''phase dictates the source of this terminal parameter
            Hmemory: Hmemory_obj
            phase:dict
            ----return parameters:list
        '''
        #source = self.category_dict['source']
        typ = 'Dream'
        if phase['dream_init'] == 'critic':
            typ = 'Critic'
        elif phase['dream_init'] == 'self':
            typ = 'Dream'
        super().forward_fn_set(Hmemory, phase, True, typ=typ)
        return self.get_parameter_values(as_list=True)

class InputEncoderTerminal(InputTerminal):
    def set_routing_masks(self):
        r_masks = get_routing_masks(
            self.settings,
            cell_types = ['Cell', 'Terminal'], #it cannot be killed
            types = [self.category_dict['type']],
            sources = [self.category_dict['source']],
            devices = self.device
        )
        self.r_masks = r_masks
    def _default_method_fn(self, param_name:str, counter = False):
        ''' in case the parameter is captured in another parameter's method
        '''
        if counter:
            def method_fn(self, Hmemory, phase:dict):
                cat = self.category_dict
                param = Hmemory.get_terminal_parameters(
                    [param_name], 'Output', (cat['type'], cat['source']),
                    self.device
                )
                self.parameters[param_name] = param
        else:
            def method_fn(self, Hmemory, phase:dict):
                return self.parameters[param_name]
        return method_fn

    def get_media(self, io_parameters, Hmemory, phase):
        '''gets the values from the enviroment be real or virtual
        and sets this values to the parameters, it uses counter parameters
        if there's not a media function
        io_parameters: dict: parameters of the counter io, ('Output')
        Hmemory: Hmemory_obj
        phase: dict
        '''      
        media = self.take_media_from_parameters(io_parameters, Hmemory, phase)
        return media

    def take_media_from_parameters(self, io_parameters, Hmemory, phase):
        '''complementary method of terminal.get_media
        ----return parameters:dict
        '''
        def tmp(name):
            sett = self.parameters_settings[name]
            param = sett['method_fn'](Hmemory, phase)
            #if not self.slave:
            #    param = torch.zeros_like(param)
            if self.imitate_master:
                imit_param = self.load_parameter_values(
                    extra='master', 
                )
                seq_dim = sett[name]['sequential']
                batch_dim = 1 if seq_dim > -1 else 0
                param = torch.cat(
                    [param, imit_param], dim = batch_dim)
            return param
        return self._param_function_return(p_func=tmp)
        
    def forward_fn(self, Hmemory, phase):
        '''takes the input of the real/dream or virtual counter terminals
            if dreaming, uses old interpretations sometimes
            Hmemory: Hmemory_obj
            phase:dict
            ----return parameters:list
        '''
        source = self.category_dict['source']
        if phase['wake']:
            '''the same as terminal.forward, also gets the parameters 
            with media fn and returns parameters in list form
            ----return parameters:list
            '''
            super().forward_fn_set(Hmemory, phase, False)
            io_parameters = self.get_parameter_values() #in case there's a parameter without media
            present = self.get_media(io_parameters, Hmemory, phase)
            past = self.extra_parameters['past']
            self.extra_parameters['past'] = present
        else:
            typ = 'Dream'
            dream_enc = phase['dream_enc']
            if dream_enc == 'critic':
                super().forward_fn(self, Hmemory, phase)
            elif dream_enc == 'dream':
                past = Hmemory.get_terminal_parameters(
                    self.param_names, 'Input', 
                    (typ , source), self.device
                )
            elif dream_enc == 'noise':
                past = self.generate_random_parameter_values()
            #elif phase['enc_past'] == 'sleep_past':
            elif dream_enc  == 'old':
                critic = Hmemory.get_terminal(
                    'Output', ('Critic', source)
                )
                old = critic.extra_parameters['old_parameters']
                if len(old) > 0:
                    past = random.choice(old)
                    past = {x:y.to(device=self.device) for x,y in past.items()}
                else: #safe choice if not past parameters
                    past = Hmemory.get_terminal_parameters(
                        self.param_names, 'Input', 
                        (typ , source), self.device
                    )
        self.set_parameter_values(past)
        return self.get_parameter_values(as_list=True)
    
class InputCriticTerminal(InputEncoderTerminal):
    pass

class InputDreamEncoderTerminal(InputEncoderTerminal):
    def forward_fn(self, Hmemory, phase):
        '''takes the input of the real/dream or virtual counter terminals
            if dreaming, uses old interpretations sometimes
            Hmemory: Hmemory_obj
            phase:dict
            ----return parameters:list
        '''
        source = self.category_dict['source']
        typ = 'Dream'
        past = Hmemory.get_terminal_parameters(
            self.param_names, 'Input', (typ, source), self.device
        )
        self.set_parameter_values(past)
        return self.get_parameter_values(as_list=True)

InputDreamCriticTerminal = InputDreamEncoderTerminal

################################################################################
#---------------------------- SPECIAL TERMINALS ---------------------------------------                                
################################################################################