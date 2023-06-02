#import torch.nn as nn
from copy import deepcopy
import random
import os
import time
import math
import shutil
import torch
import torch.nn.functional as F
import numpy as np

#from torch._C import device
from generators.models import MLP, CellMLP
from generators.cellmodels import CellModel
from generators.extra_models.utils import set_learn_functions, str2act
import utils


class Cell(object):
    '''
    methods for saving and loading the model it contains, also statistics about
    diffusion categories, time used and frequency

    name:str: name of the cell, binary code or category if terminal
    mainfolder:str: folder to save cell's checkpoint
    CellModelClass: nn.Module: class of subactions models
    MainModelClass: nn.Module: class of main model
    init_behaviour: str: set of actions at initialization level
        'set', sets the model
        'set_and_load', sets the model and loads the checkpoint from disk
        'set_and_save', sets the model and saves it if it doesn't exist
        'reset_and_save', deletes from disk, sets and saves
        'expand_and_save', loads, expands, sets and saves
    device:torch.device:
    settings:dict:
    record_statistics:bool:makes the cell record statistics
    n_cells:int: division of the cell, makes the main values divide
        by this number and those values then are forwarded with independent
        replicas and subactions
    lr: float: learning rate
    loss_fn: str: loss function
    optimizer_type:str:
    decay:float:weight decay
    model_settings:dict: settings of main model
    cell_model_settings:dict: settings of subhead model
    detach_stbk: bool: detaches some values to reduce ram usage
        and is not that important that these values are connected
        to the main tree
    '''
    
    def __init__(self,
            name = '',
            mainfolder = './',
            diff_index = 0,
            CellModelClass = CellModel,
            MainModelClass = CellMLP,
            init_behaviour = 'set_and_load',
            device = None, #The device is set by the memory, and subsecuentely by the terminal
            settings = None, #EXPOSED ARGS -----
            lr = 0.0002,
            loss_fn = 'MSE',
            optimizer_type = 'SGDmom',
            decay = 0.1,
            terminals = None,
            model_settings = None,
            cell_model_settings = None,
            n_routings = None,
            detach_stbk = True,
            #sample = {}
        ):
        if settings is None:
            settings = {}
        if device is None:
            device = torch.device('cpu')
        if model_settings is None:
            model_settings = {}
        if cell_model_settings is None:
            cell_model_settings = {}
        if n_routings is None:
            n_routings = settings[
                'actions']['where']['routings']

        self.name = name
        self.mainfolder = mainfolder
        self.folder_path = os.path.join(mainfolder, name)
        self.chk_path = os.path.join(
            self.folder_path, settings['memory']['checkpoint_path']
        )
        self.type = 'cell'
        self.io_index = None

        self.CellModelClass = CellModelClass
        self.MainModelClass = MainModelClass
        self.device = device
        self.settings = settings
        self.model_settings = model_settings
        self.cell_model_settings = cell_model_settings
        # use sample code
        self.deepness = 0
        self.branching = 0
        self.lr = lr
        self.decay = decay
        self.dyn_factor = 1.0
        # self.n_routing = settings['subactions']['where']['n_routings']
        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type
        self.where_act = {}
        self.batch_size = 1

        self.train = True
        self.to_back = []

        self.cell_values = None
        # self.cycle_execution = settings['cycle_execution']
        self.exp_t0 = settings['exp_t0']
        self.exp_xt = 0.0
        self.not_used_cnt = 0 #consecutive
        self.used_cnt = 0
        self.model = None
        self.diff_die = None #diffuseness when dying
        self.traversed = 0
        self.cycle_notraversed = 0

        self.terminals = terminals
        self.last_time = time.time()
        self.current_connections = []
        self.connections_masks = {}
        self.connections_diff_index = None

        self.diff_index = diff_index
        self.cycle_input = {}
        self.cycle_values = []
        self.cycle_reward = 0 #collected cycle reward, added to the terminal
        #at the end
        self.cell_values = None
        self.cell_input = None
        self.n_input = 1
        #defined in parented class
        self.counts = torch.zeros(
            self.settings['n_cells'] + len(self.settings[
                'terminal_io'])*self.settings['diffuseness_dim'],
            device=device
        ) #cells, io*term
        self.cell_back_routing = []
        if detach_stbk:
            self.dtbk = lambda x: x.detach()
        else:
            self.dtbk = lambda x: x

        self.model_init(init_behaviour)
        '''
        if not shared_state == None:
            self.wait_for_loss()
        '''
    def model_init(self, init_behaviour):
        '''Init action to take, after device has been defined
            init_behaviour:str
            set: creates model and statistics
            load: loads model and statistics from file
            save: saves models and statistics to file
            delete: deletes files with folder
            expand: expands capacity model by adding layers
            mutate: modifies connections to the connections of 
                    best similar cell
            mutate_cell: modifies weights to the best similar
                    cell
            combinations:
            set_save: resets model and statistics


        '''
        checkpoint = None
        if 'load' in init_behaviour:
            checkpoint = self.get_checkpoint()
        if 'set' in init_behaviour:            
            self.set_model()
            self.load_state(checkpoint)
        if 'delete' in init_behaviour:
            if os.path.exists(self.folder_path): 
                shutil.rmtree(self.folder_path)
        if 'mutate' in init_behaviour:
            self.mutate_connections()
        if 'mutate_cell' in init_behaviour:
            pass
            #self.mutate_model()
        if 'expand' in init_behaviour:
            self.expand_model()
        if 'save' in init_behaviour:
            if not os.path.exists(self.chk_path):
                if not os.path.exists(self.folder_folder):
                    utils.check_make_folder(self.folder_path)
            else:
                print('Model for ' + self.name+ ' already exists')
            self.save_state()
        
    def declare_model(self,):
        '''returns cell model
            ----return nn.Module
        '''
        return self.CellModelClass(
            ModelClass = self.MainModelClass, model_settings = self.model_settings, 
            settings = self.settings, device= self.device,
            **self.cell_model_settings
            ).to(self.device)

    def set_model(self,):
        '''Initiliazes the model with self.ModelClass, with the optimizer and loss fuction
        Also sets the size of the vector that goes in the input and the output by traversion
        of the chain.
        '''
        print('Setting model for ' + self.name)
        self.model = self.declare_model()
        self.model.train()
        self.generic_input_fn = lambda x: [torch.sum(
            torch.stack(x), dim = 0)]
        self.reset_cycle_values()
        self.input_fn = {
            'diffuseness': self.generic_input_fn,
            'maturity': self.generic_input_fn,
        }
        if self.train:
            optimizer, _ = self.get_learn_functions()
            self.optimizer = optimizer
        else:
            self.optimizer = None
        #self.out_len = self.model.out_len
        #self.in_len = self.model.in_len
        
    def expand_model(self):
        '''increases actions weight dimensions, in subaction settings'''
        checkpoint = self.get_checkpoint()
        model_state_dict = checkpoint['model_state_dict']
        for subact in self.model.use_subactions:
            hset = self.settings['subactions'][subact]
            sact_preset = self.settings['subactions_preset'][subact]
            for h, psett in enumerate(sact_preset):
                key = '.'.join([
                    subact + '_models',
                    'heads', str(h), 'model'])
                max_i = 0
                for i in range(1000):
                    bias_key = '.'.join([key, str(i), 'bias'])
                    if bias_key in model_state_dict:
                        max_i = max(i, max_i)
                bias_key = '.'.join([key, str(max_i), 'bias'])
                weight_key = '.'.join([key, str(max_i), 'weight'])
                dim_d = psett['out_dim']

                bias = model_state_dict[bias_key]
                weight = model_state_dict[weight_key]

                ddim = dim_d - bias.size()[0]
                if ddim > 0:
                    extra_t = torch.zeros(ddim).uniform_(
                        bias.min(), bias.max()
                    )
                    bias = torch.cat([bias, extra_t], 0)
                    pd = weight.size()[1]
                    extra_t = torch.zeros(ddim, pd).uniform_(
                        bias.min(), bias.max()
                    )
                    weight = torch.cat([weight, extra_t], 0)
                    model_state_dict[bias_key] = bias
                    model_state_dict[weight_key] = weight
        checkpoint['statistics']['maturity'] *= 0.5 #expansion makes the cell less mature
        checkpoint['statistics']['n'] *= 0.5
        self.load_state(checkpoint)

    def create_statistics(self,):
        '''Creates dict with statistics about traversions in the cell, like avg loss
            avg diffuseness etc
            ----return dict
        '''
        diff_dim = self.settings['diffuseness_dims']['Output']
        #assumes the number of Input and Output terminals are the same
        stats = { 
            'diffuseness':F.normalize(torch.rand(
                2, diff_dim), dim=1), #average of graph diffuseness that traverse this cell
            'loss': 10 + random.random()*0.1, #average of losses = avg(loss), big initilized
            #'rate':0, #Rate at which is used this cell, n_t/avgreltime
            'abstime': time.time(), #updated every time the cell is traversed with time.time()
            'avgreltime': 1, # average relative time = avg(time.time()-abstime)
            'n':1, #times the bead has been traversed
            'reward_n': 1,
            #'n_t': [], #the same as n, this is over a range of time
            'avg_deepness': 0, #average deepness, used for extra lr adjustment lr*0.75^avg_deepness
            'maturity': 0,
            'mutated': 0,
            'connections':[
                { 'avg_loss':1.0, 'n': 1, 
                   'cell_type': 'cell', 'where': self.name,
                    'source': None, 'type': None,
                }
                #n_kills, depends on number of routings
                #dim/2_cells
                #dim/2_outputs
            ]
        }
        self.init_connections()
        return stats

    def get_connection_masks(self, ):
        ''' Converts the connections descriptions into
            masks that can be multiplied with incoming mask
        '''
        all_types = self.settings['all_types']
        all_sources = self.settings['all_sources']
        conn_masks = self.settings['conn_masks']
        conn_masks = {
            i:conn.detach().clone().to(device=self.device) 
            for (i, conn) in conn_masks.items()
        }
        for i, conn in enumerate(self.statistics['connections']):
            if conn['cell_type'] == 'cell':
                conn_masks['cell_type'][i][0] = 1.0
            elif conn['cell_type'] == 'terminal':
                conn_masks['cell_type'][i][1] = 0.0
                indx = all_types.index(conn['type'])
                conn_masks['type'][i][indx] = 1.0
                indx = all_sources.index(conn['source'])
                conn_masks['source'][i][indx] = 1.0
            elif conn['cell_type'] == 'kill':
                conn_masks['cell_type'][i][2] = 1.0
            else:
                raise ValueError(
                    'Cell type not recognized' + conn['cell_type']
                )
        return conn_masks

    def init_connections(self,):
        ''' Randomly initialize the connections in cells
        '''
        terminals = self.terminals
        n_conn = self.model.n_connections
        routings = self.model.n_routings
        output_terminals = terminals['Output']
        n_terminals = len(output_terminals)
        n_cells = self.settings['n_cells']
        n_receptors = self.model.n_receptors
        ratio = self.settings['cell_limits']['mutation']['cell_ratio']
        conn_dict = {}
        eff_n_conn = n_conn - routings #kill routings
        
        conn_dict['cell_type'] = ['Cell']*math.floor(eff_n_conn*ratio)
        conn_dict['cell_type'] += ['Terminal']*math.ceil(eff_n_conn*(1-ratio))
        conn_dict['types'] = [None]*n_conn
        conn_dict['sources'] = [None]*n_conn
        conn_dict['where'] = [None]*n_conn
        conn_dict['receptor'] = [None]*n_conn
        conn_dict['diff_index'] = [None]*n_conn
        tsock = self.settings['terminal_sockets']

        for i in range(eff_n_conn):
            if conn_dict['cell_type'] == 'Cell':
                celln = random.randint(0, n_cells) #int
                conn_dict['where'][i] = bin(celln)[2:] #int -> binary
                conn_dict['diff_index'][i] = int(celln)
                #no receptor for now, here could be receptors if there are
                #cells with different dimentions

            elif conn_dict['cell_type'] == 'Terminal':
                termn = random.randint(0, n_terminals-1) #-1 for 0-indexing, and -1 for Kill terminal
                terminal = output_terminals[termn]
                source = terminal.category_dict['source']
                conn_dict['sources'][i] = source
                conn_dict['types'][i] = terminal.category_dict['type']
                conn_dict['where'][i] = terminal.category
                conn_dict['diff_index'][i] = int(n_cells + terminal.index)
                if 'receptors' in tsock[source]: #overrides generic amount of receptors
                    recepn = random.randint(0, tsock[source]['receptors']-1)
                else:
                    recepn = random.randint(0, n_receptors-1)
                conn_dict['receptor'][i] = recepn

        conn_dict['cell_type'] += ['Kill']*routings #kill terminal
        conn_dict['sources'][-routings:] = ['Kill']*routings
        conn_dict['types'][-routings:] = ['Kill']*routings
        conn_dict['receptor'][-routings:] = [None]*routings
        conn_dict['diff_index'][-routings:] = [-1]*routings #last will be norm
        #vector of 1/diff_dim values
        #conn_dict['where'][routings] = self.name #self routing
        connections = []
        for i in range(n_conn):
            conn = {
                'avg_loss': 1.0, 
                'n': 1,
                'where': conn_dict['where'][i],
                'source': conn_dict['sources'][i],
                'type': conn_dict['types'][i],
                'receptor': conn_dict['receptor'][i],
                'diff_index': conn_dict['diff_index'][i]
            }
            connections.append(conn)
        self.statistics['connections'] = connections

    def set_diff_index_connections(self, ):
        '''gets and sets the diff indices of all connections, so it only does it once'''
        diff_indeces = []
        for conn in self.statistics['connections']:
            diff_indeces.append(conn['diff_index'])
        self.diff_indeces = torch.tensor(self.diff_indeces, device = self.device)
        self.same_diff_indeces = self.diff_index == self.diff_indeces

    def mutate_connections(self, Hmemory):
        ''' mutates the connections that have the worst loss and
            the least used (fitness)
            Hmemory: Hmemory class obj
        '''
        connections = self.statistics['connections']
        n_thr = self.settings['n_threshold']
        mut_limit = max(
            int(self.settings['cell_limits']['mutation']*len(connections)),
            1
        )
        cell_ratio = self.settings['cell_limits']['cell_ratio']
        for conn in connections:
            conn['fitness'] = min(conn['n'],n_thr)/conn['avg_loss']
        connasort = sorted(connections, key=lambda x:x['fitness'])
        n_cells = self.settings['n_cells']
        n_terminals = len(Hmemory.output_terminals) - 1 
        for i, conn in enumerate(connasort):
            if i < mut_limit:
                conn['avg_loss'] = 1.0
                conn['n'] = 1
                if random.random() < cell_ratio:
                    conn['where'] = bin(random.randint(0, n_cells))[2:]
                    conn['source'] = None
                    conn['type'] = None
                else:
                    termn = random.randint(0, n_terminals)
                    terminal = Hmemory.output_terminals[termn]
                    conn['where'] = terminal.category
                    conn['source'] = terminal.category_dict['source']
                    conn['type'] = terminal.category_dict['type']
            else:
                break
    def update_diffuseness(self, fractions, stats, n):
        ''' update diffuseness of input and output
            fractions: cell, io, term
        '''
        fractions = fractions.to(self.device)
        diffuseness = fractions[self.diff_index] # io, term
        stats['diffuseness'] = F.normalize(utils.dynamic_average_pyt(
            stats['diffuseness'], diffuseness, stats['n']
        ), dim = 1)

    def update_statistics(self, now_t, loss, fractions):
        ''' Updates the statistics of the cell for later pool operations: mutate cells
        now_t: time these statistics were updated
        diff: diffusion of chain
        loss: loss of the critic loss, encoder loss is not recorded
        ----return statistics:dict
        '''
        if self.train:
            stats = self.statistics
            n = stats['n']
            stats['n'] = n + 1*(n < self.settings['n_threshold'])
            reltime = now_t - stats['abstime']
            stats['avgreltime'] = utils.dynamic_average(
                stats['avgreltime'], reltime, n)
            self.update_diffuseness(fractions, stats, n)
            stats['loss'] = utils.dynamic_average(
                stats['loss'], min(loss.item(), 1e4), n
            )
            stats['abstime'] = now_t
            stats['maturity'] = min(1, stats['n']/(
                self.settings['n_threshold'])/(stats['loss']+0.1))
            for conn in set(self.current_connections):
                connsta = self.statistics['connections'][conn]
                connsta['loss'] = utils.dynamic_average(connsta['loss'], loss)
                connsta['n'] += 1
        return self.statistics


    def get_learn_functions(self,):
        '''Initializes the optimizer, useful for reseting the optimizer when a new stage of
        learing is going on (when the cell stops learning this is initialized)
        ----return (optimizer:nn.optim, loss function:func)
        '''
        return set_learn_functions(
            optimizer_type = self.optimizer_type, 
            lf_type = self.loss_fn, 
            lr = self.lr,
            model=self.model,
            decay=self.decay,
        )
    def get_checkpoint(self):
        '''gets checkpoint from files
            ----return checkpoint:dict
        '''
        if os.path.exists(self.chk_path):
            return torch.load(self.chk_path, map_location=self.device)
        else:
            return None

    def load_state(self, checkpoint = None):
        '''loads a checkpoint of the model. If there's no statistics then it creates one.
            checkpoint:dict: it can load the checkpoint stored in
            CPU
            ----return checkpoint:dict
        '''
        if checkpoint is None:
            self.statistics = self.create_statistics()
            self.connections_masks = self.get_connection_masks()
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.statistics = checkpoint['statistics']
            self.statistics['diffuseness'] = self.statistics[
                'diffuseness'].to(self.device)
        self.set_diff_index_connections()
        return checkpoint

    def get_state(self):
        '''gets the state of the cell (a checkpoint if saved)
        '''
        state = {
            #'sample': self.sample,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'statistics': self.statistics
        }
        return state

    def save_state(self):
        '''saves a checkpoint of the model'''
        if self.train:
            torch.save(self.get_state(), self.chk_path)
        
    def delete_cell(self,):
        '''deletes the Cell with the model and statistics'''
        shutil.rmtree(self.folder_path)
        
    def copy2id(self, new_id):
        shutil.copytree(self.folder_path, os.path.join(self.mainfolder,new_id))

    def diminish_parameters(self,):
        ''' This is the forgetting method, it reduces the weights and biases
        of the model so it forgets old patterns
        TODO, make the forgetting method use instead a glorot inizialization
        or such instead of 0.1
        '''
        if random.random() < self.lr*self.traversed:
            model = self.model
            for module in model.modules():
                parameters = module.parameters()
                for param in parameters:
                    param.mul_(1-self.lr)
                    param.add_(torch.rand_like(param)*self.lr)

    def get_unique_name(self):
        '''unique name'''
        return self.name

    def reset_cycle_values(self,):
        '''resets cycle values and inputs'''
        self.cycle_values = [[]]*self.model.n_receptors
        self.cycle_input = {}

    def set_cell_entry(self, recep, values, diff_index, 
            counts, cell_back_routing, cycle_reward,
            **kwargs):
        ''' cell:obj, 
            recep: receptor index,
            values: values to traverse latter,
        '''
        counts = counts.clone()
        counts[diff_index] += 1
        self.cycle_values[recep].append(values) #(recep), (N), B, L
        #self.counts = self.counts + counts > 1 #for output
        self.counts = self.counts + counts
        self.cell_back_routing += list(set(cell_back_routing))
        self.cycle_reward += cycle_reward
        for key,val in kwargs:
            self.cycle_input[key] = val
        
    def get_cell_entries(self,):
        ''' processes all the inputs recieved during cycle
        '''
        cell_values= self.model.receptor(
            self.cycle_values) #(recep),(N),B, L -> N, B, L
        self.cell_values = self.model.filtering_model(
            cell_values) #N, B, L
        cs = cell_values.size()
        self.n_input = cs[0]
        self.batch_size = cs[1]

        cell_input = {}
        for k,_ in self.cycle_input.items():
            val = self.input_fn[k](
                self.cycle_input[k]) # (1), L
            cell_input[k] = val
        self.cell_input = cell_input
        #self.update_diff_count()
        self.reset_cycle_values()

    def pre_traverse(self, values, now_t):
        '''
        set the internal values, load and set the residue, also load
        and write to memory, also filters by predicting and substracting
        main values with internal valaues tensor
        values: tensor: # B, Lat
        internal_terminal: tensor: values from internal terminal
        init_time: float: time since chain was initiated
        traverse_enconding: tensor: past traverse encoding, deepness, in_terminal order, branching
        diffuseness_values: tensor: source, type and IO of past diffuseness
        device_resources: float: fraction of GPU already taken
        terminal_order: float: from 0 to 1, 1/n_terminals
        last_cell_maturity: maturity of last cell, proportional to
            the number of traversions and inversely to the loss
        tick_encoding: list: n-modulo number of global iterations
        ----return: tensor
        '''
        values = self.model.filtering_model(values) #B, Lat or B*N, Lat
        values = self.model.memory_model(
            values, now_t, #B, Lat; float
            self.statistics['maturity'], #float
            self.statistics['n'] #int
        )
        return values #B, Lat
        
    def short_inner_traverse(self, values, now_t):
        '''only traverses the model
            values: B, lat
        '''
        return self.model(values, now_t, self.statistics['n'])

    def inner_traverse(self, values, now_t, r_mask, r_matrix, out_diff, penalty):
        '''
        values are passed through the main cell model, a latent model, this is then
        passed to the where and what models, the what models first, it sets the 
        actions to take, where or output for all the routings
        values: tensor: with values preprocessed in the pretraverse phase
        now_t: float: time
        r_masks: list of dict of tensors: tensor masks for directed routing, it masks
            connections that are incompatible with the terminal that spawn the
            terminal/phase, if the cell has one receptor
            then is terminal's routing masks, otherwise is the phase mask only, it
            has to pass the mask with the correct device also.
            (N), {kmask}, conn, dl
        p_conn: penalized connections for not being traversed enough
        ----return (main_values:tensor,
            where_list: list: actions to take, like routing or expressing)
        '''
        
        main_values = self.short_inner_traverse(values, now_t) #B(*N), L
        if self.model.n_reward_contexts > 0:
            rwd_pairs, qlat = self.model.reward_model(
                main_values, self.statistics['n'])
            main_values = qlat
            for pair in rwd_pairs:
                self.to_back.append([
                    *pair,
                    'reward'
                ])
            self.cycle_reward += torch.sum(rwd_pairs[0][1], dim=0) #term
        hist = self.model.main_model.get_buffer('hist')[1]
        self.to_back.append([
            utils.curiosity_value(hist),
            self.dtbk(main_values),
            'curiosity']
        )
        where_act = {} #[values_to_route, connections, indeces], each with list

        if 'where' in self.model.use_actions:
            where_model = self.model.where_model
            cell_routings = self.model.n_routings #scalar
            where_values = where_model(main_values) #B(*N), conn
            #----------------------DORMANT CONNECTIONS (ANTI)PENALTY---------
            if torch.any(p_conn > 0.01):
                p_conn = penalty[self.diff_indeces]
                counts_conn = self.counts[self.diff_indeces]
                pair = where_model.penalty_toback(
                    self.dtbk(where_values), p_conn, counts_conn
                )
                self.to_back.append([
                    *pair,
                    'regression'
                ])
            #-----------------------DIFFUSE STRENGTH MATRIX--------------
            in_diff = self.statistics['diffuseness'][0] #input diffuseness
            #exp_n = self.exp_xt
            out_diff_local = out_diff[self.diff_indeces] #(conn), term
            out_diff_local = torch.stack(out_diff_local) #conn, term
            in_diff = self.statistics['diffuseness'][0] #term
            diff_product = torch.einsum('co, i -> cio', out_diff_local, in_diff) #conn, term, term
            if torch.any(self.same_diff_indeces):
                diff_product = (1 - self.same_diff_indeces)*diff_product #conn, term, term
                diff_product += self.same_diff_indeces*torch.ones_like(diff_product)
            diff_product = r_matrix * diff_product  #conn, term, term
            diff_conn = torch.sum(diff_product, dim=(1,2)) + 1e-5 #conn
            #-----------------------PHASE MASK VECTOR-----------------
            conn_masks = self.connections_masks #cell_type, type, source
            #any for any descriptor, all descriptors for all descriptors
            bmask = [torch.any( #(k), conn, dl -> (k), conn, dl
            r_mask[k]*conn_masks[k] > 0.5, dim=1) for k in conn_masks.keys()]
            bmask = torch.stack(bmask, dim=0) # (k), conn -> k, conn
            bmask = torch.all(bmask, dim=0) # conn
            bmask = bmask * diff_conn #conn

            where_masked = where_values*bmask # B(*N), conn
            #n_shape = (self.n_input, self.batch_size, where_values.size()[1])
            to_route_val, indeces = torch.topk(where_masked.detach(),
                k = cell_routings, dim=2) # B(*N), R ; # B(*N), R, (int)
            to_route_val = to_route_val.view(-1) # B*N*R
            indices = indices.view(-1) #B*N*R
            where_act['to_route_values'] = to_route_val
            for indx in indeces:
                conn = self.statistics.connections[indx]
                where_act['connections'].append(conn)
            where_act['indices'] = indeces
            self.current_connections.append(indeces)
            self.where_act = where_act
        return main_values

    def cell_reward_train(self, rewards, output_fractions):
        ''' use diffuseness of cycle at the end
            has to be trained in terminal
            reward: term, it could be various rewards at the same time
            output_fractions: term, seq, cell + 2*term
        '''
        act_disc = output_fractions[:, :, self.diff_index] # term, seq
        drwd = rewards * act_disc # term, seq
        hist = self.model.main_model.get_buffer('hist')[1] # seq, maindim
        train_q = self.model.reward_model.train(
            hist, drwd, self.statistics['n'] )
        loss = F.l1_loss(train_q[0], train_q[1])
        return loss

    def set_lr(self, ext_factor = 1.0):
        '''
        modifies the lr of the optimizer
        dyn_factor:float: that does scalar multiplication of the base learning rate, self.lr
        '''
        return None
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = dyn_lr

    def set_weight_decay(self):
        '''weight decay is high early on then it becomes lower
        '''
        for param_group in self.optimizer.param_groups:
            param_group[
                'weight_decay'] = (1-self.statistics['maturity'])*0.5
    
    def extra_traverse(self, values, output_prev):
        '''does the residual sum of the input given to the cell, in pretraverse
            and normalization
            values: tensor: main values
            output_prev: tensor: values before pretraverse
            ----return tensor
        '''
        values = values + output_prev
        values = self.model.norm_layer(values)
        return values
    def clean_next_iteration(self):
        pass

    def clean_next(self,):
        '''cleans cell for next cycle'''
        for k in self.cell_input.keys():
            self.cell_input[k] = [torch.zeros_like(k)]
        self.counts = torch.zeros_like(self.counts)
        self.to_back = []
        self.current_connections = []
        self.diff_die = None
        self.diff_loss = 0
        self.traversed = 0
        self.model.clean_next()