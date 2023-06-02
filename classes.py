import torch
import os
import terminals.container as term_classes
from generators.bead import Bead, BeadMemory, OffCell
import inspect
import utils
import random
import math

class CellHierarchicalMemoryManager():
    '''
        max items in memory, max items in gpu, max items in SSD, max items in HDD
        it deals with all memory functions, loading and unloading, saving, syncing
        devices' cells with each other, etc. also stores terminals and defines
        special terminals like the Kill terminal

        term_obj: dict: terminal objects defined in a call of get_terminals
        main_device: str: maindevice chosen
        acc_devices: dict: acceleration devices
        dev_order: list: order of the devices
        settings: dict:
    '''
    def __init__(self, 
            main_device='cpu',
            phase = {},
            acc_devices={},
            dev_order = [],
            init_behaviour = 'set_and_load',
            settings = {}):

        self.main_input_terminals = []
        self.imitate_input_terminals = []
        self.unique_terminals = {}
        self.devices = settings['memory']['devices']
        self.acc_devices = acc_devices
        self.dev_order = dev_order
        self.main_device = main_device
        self.settings = settings
        self.mode = None
        self.init_behaviour = init_behaviour
        
        self.cpucells = {}
        self.phase = phase
        self.wake = phase['wake']
        self.dev_dist = self.get_device_distribution()
        self.cells = {}
        self.cells_list = []
        self.cells_device = {x:{} for x in self.acc_devices}
        self.cells_online = {x:{} for x in self.acc_devices}
        self.cell_mdevice = {} #main device
        self.cell_location = {} #
        self.term_reward_track = []
        self.cell_terminals = []

        self.experiments_folder = os.path.join(
            settings['memory']['experiments_folder'],
            settings['memory']['experiment'],
        )
        self.cycles = 0
        self.used_cnt = settings['memory']['used_cnt']
        self.nouse_cnt = settings['memory']['nouse_cnt']
        self.tflr_freq = settings['memory']['filter_time_freq']
        self.save_freq = settings['memory']['update_checkpoints']
        #used for system sensors
        self.update_tickings()
        self.last_loss = torch.zeros(1, device=main_device)
        self.last_routings = 1000
        self.last_cycle_time = 1.0

        self.mloss_p = settings['memory']['mem_loss_power']
        mlsl = settings['memory']['mem_loss_top']
        self.mloss_t = 1-(1/self.mloss_p)**(mlsl/(mlsl+1))
    
    def set_main(self, is_slave = True):
        self.slave_terminals = True
        relation_path = 'slave' if is_slave else 'master'
        self.bead_folder = os.path.join(
            self.experiments_folder,
            relation_path,
            'beads'
        )
        self.terminal_folder = os.path.join(
            self.experiments_folder,
            relation_path,
            'terminals'
        )
        self.set_terminals() #also sets inital rewards
        if is_slave:
            self.set_all_beads()

    def get_routing_masks(self, ):
        ''' gets routing masks taking the types allowed in each wakephase'''
        term_types = self.settings['phase_terminal_types'][self.wake]
        devices = [*self.dev_dist['terminals']['Input'], *self.dev_dist['cells']]
        r_masks = utils.get_routing_masks(
            self.settings,
            types = term_types,
            devices = devices
        )
        r_matrix = {dev:self.settings[
            'routing_strength_matrix'].detach().clone(
            ).to(dev) for dev in devices}
        return r_masks, r_matrix

    def get_device_distribution(self):
        '''
            gets the devices distribution, how many and which ones go to each terminal
            and the cells
            ----return dev_dist:dict:has terminals with io and cells
        '''
        ddist = self.settings['memory']['devices_distribution']
        dev_dist = {
            'terminals':{
                io:[self.main_device] for io in self.settings['terminal_io']},
            'cells':[self.main_device]
        }
        if len(self.dev_order) > 1:
            if ddist == 'split':
                dev_dist['terminals'] = {
                    io:self.dev_order for io in self.settings['terminal_io']}
                dev_dist['cells'] = self.dev_order
            elif ddist == 'iocells': #default
                if len(self.dev_order) == 2: #input/output same device
                    dev_dist['cells'] = [self.dev_order[1]]
                elif len(self.dev_order) > 2: #input/output different device, the rest to cells
                    dev_dist['terminals']['Output'] = [self.dev_order[1]]
                    dev_dist['cells'] = self.dev_order[2:]
        return dev_dist

    def set_all_beads(self):
        '''Sets all the beads with the respective mode
        '''
        utils.check_make_folder(self.bead_folder)
        hset = self.settings['subactions']['where']
        key = hset['key_index']['bin_routing']
        routing_dim = hset['preset_used'][key]['out_dim']
        len_beads = 2**routing_dim
        cdev = self.acc_devices['cells']
        for i in range(len_beads):
            uname = bin(i)[2:]
            device = cdev[len(cdev)%i]
            bead = self.create_bead(
                uname, device, self.init_behaviour
            )
            self.cells[uname] = bead
            self.cells_list.append(bead)
            self.cell_mdevice[device][uname] = bead
            if self.mode == 'expand':
                if not os.path.exists(bead.chk_path):
                    for k in range(routing_dim):
                        chk_path = os.path.join(
                            self.bead_folder, uname[:-k], 'checkpoint.pth'
                        )
                        if os.path.exists(chk_path):
                            break
                    bead.chk_path = chk_path
                bead.model_init('expand_and_save')
            if self.mode == 'mutate':
                self.routing_dim = routing_dim #check later
                
    def init_terminals(self, ):
        '''
            Loads the classes for preloading the objects terminals, and the models.
            This is passed later to the memory where terminals are stored there.
            uses settings expansion done in postprocessing.
            settings: dict
            ---returns:(
                    obj_terminals: dict: key = io, values = dict: 
                        key = terminalname (category), values = terminal object
                    terminal_classes: the same as above but with terminal class
                )
        '''
        obj_terminals = {}
        terminal_classes = {}
        dev_dist = self.dev_dist['terminals']
        settings = self.settings
         #used for reward in cells
        #otherwise it could have outputs without any update
        reward_types_track = self.settings['types_reward_track']
        is_slave = self.slave_terminals

        for io in settings['terminal_io']:
            obj_terminals[io] = []
            len_devices = len(dev_dist[io])
            for i, ttup in enumerate(settings['terminal_expansion'][io]):
                fulli = settings['terminal_expansion'][io].index(ttup)

                classes_found = 0
                sch = [io, ttup[1]]
                
                for k in range(3):
                    str_k = sch[:k]
                    sclass = ''.join([ttup[0],*str_k])
                    tclass = ''.join([*str_k,'Terminal'])
                    if hasattr(term_classes, sclass):
                        SourceClass = getattr(term_classes, sclass)
                        classes_found += 1
                    if hasattr(term_classes, tclass):
                        TerminalClass = getattr(term_classes, tclass)
                        classes_found += 1
                if classes_found < 2:
                    raise Exception('Classes not found for:', sclass, tclass)

                class clss(SourceClass, TerminalClass):
                    slave = is_slave
                    index = i
                    full_index = fulli
                    number = ttup[3]
                    mainfolder = self.terminal_folder
                    term_settings = settings['terminal_sockets'][ttup[0]]
                    category_dict = {
                        'source':ttup[2], #source name
                        'type':sch[1], #type capitalized
                        'io': sch[0], #io capitalized
                    }
                    device = torch.device(
                        device=dev_dist[io][ttup[3]%len_devices]
                    )
                    def __init__(self, settings):
                        super().__init__(settings)
                        assert len(self.param_names)>0, "No parameters in "+self.category
                        #print(clss.__mro__)

                term_settings = clss.term_settings
                
                if (self.slave_terminals and io=='Output') or io == 'Input': 
                    #only set input terminals for master
                    #cos master doesn't learn.
                    terminal = clss(settings)
                    if not utils.check_key(term_settings, 'only_initilize'):
                        obj_terminals[io].append(
                            terminal
                        )

                if terminal.shared_category != terminal.category:
                    if io == 'Input':
                        self.main_input_terminals.append(terminal)
                        if terminal.imitate_master:
                            self.imitate_input_terminals.append(terminal)

                if 'reward' in terminal.param_names:
                    source = terminal.category_dict['source']
                    if terminal.category_dict['type'] in reward_types_track:
                        if not source in self.term_reward_track:
                            self.term_reward_track.append(source)
                    else:
                        if not source in self.term_reward_track:
                            if 'reward' in terminal.param_names:
                                terminal.param_names.remove('reward')
                    #if by accident Virtual terminal has reward parameter
                    #but no reward param in Critic or Real, then reward parameter is 
                    #removed.
                
        class clss(term_classes.KillTerminal, term_classes.OutputTerminal):
            mainfolder = self.terminal_folder
            category_dict = {
                    'source':'Kill', #source name
                    'type':'', #type capitalized
                    'io': 'Output', #io capitalized
                }
        killterminal = clss(settings)
        obj_terminals['Output'].append(killterminal)
        self.terminals = obj_terminals

    def set_terminals(self):
        '''
            sets terminal's cell and unique terminals
        '''
        self.init_terminals()
        for io, io_terms in self.terminals.items():
            for i, terminal in enumerate(io_terms):
                terminal.set_cell(self.init_behaviour, self.terminals, self.phase)
                terminal.set_diffuseness_io_values()
                terminal.restore_parameters()
                        
    def get_terminal_parameters(self, param_names = [], io='Output', 
            term_tuple = (), to_device = None, extra=None,
            terminal = None, as_list=False, strict = True):  
        '''
            gets the terminal's parameters, with some options
            param_names: list: parameters to get
            io: str:
            term_tuple: tuple: (type, term_name(socket+i) or source) 
            to_device: str OR torch.device: 
                device the parameters should have
            extra: str: extra parameter str to get from
            terminal: terminal_obj: if none loads the one in term_tuple
                and io
            as_list: bool: return parameters as a list ordered like
                in param_names or the ones in terminal
            strict: bool: get the parameters strictly in param names,
                it will error if it doesn't find it.

            ---return parameters:dict
        '''  
        if terminal == None:
            terminal = self.get_terminal(io, term_tuple)
        if not strict:
            param_names = [pn for pn in terminal.param_names if pn in param_names]
        parameters = terminal.get_parameter_values(param_names, extra, as_list)
        if as_list:
            ky = list(range(len(parameters)))
        else:
            ky = parameters.keys()
        if not to_device:
            to_device = terminal.device
        for i in ky:
            parameters[i] = parameters[i].detach().to(device=to_device)
        return parameters

    def get_terminal(self, io, term_tuple=()):
        ''' io: str: input or output
            term_tuple: tuple: (type, term_name(socket+i) or source)
        '''
        io_termexp = self.settings['terminal_expansion_fast'][io]
        if term_tuple in io_termexp:
            idx = io_termexp.index(term_tuple)
            return self.terminals[io][idx]
        else:
            return None

    def create_bead(self, uname, device = None, mode = 'set'):
        ''' Creates bead
            uname: str: name of bead in binary code
            gpu_device: str or torch.device
            mode: str: inital action to take when bead is
                created
            ----return: bead_obj
        '''
        if uname[0] == 1:
            clss = BeadMemory
        else:
            clss = Bead
        num = int(uname, 2)
        if device:
            dev = device
        else:
            celldevs = self.dev_dist['cells']
            dev = celldevs[num % len(celldevs)]
        bead = clss(
            name = uname,
            mainfolder = self.bead_folder,
            diff_index = num,
            device = dev,
            init_behaviour = mode,
            settings = self.settings,
            model_settings = self.settings['main_model_parameters'].copy(),
            terminals = self.terminals
        )
        return bead
    def get_cell(self, name, device = None):
        ''' gets cell and returns it with the device specified
        '''
        if isinstance(name, str):
            cell = self.cells[name]
        else:
            cell = self.cells_list[name]
        return cell.to(device)

    def _get_cell(self, name, device):
        '''Gets the by name, and sets it to the device requested
            1) searches in online cells first, (cells that are being used)
            2) searches in offline cells, and transfers it to online cells
            3) searches in cells in CPU
            4) searches in cells in HDD
            name:str: name of the cell in binary form
            device:str OR torch.device:
            ----return cell:cell_obj
        '''
        if name in self.cells_online[device]:
            return self.cells_online[device][name]
        elif name in self.cells[device]:
            self.cells_online[device][name] = self.cells[device][name]
            del(self.cells[device][name])
            return self.cells_online[device][name]
        else:
            #self.pull_cells(device, one = True, criteria='random')
            if name in self.cpucells:
                cell = self.create_bead(name, device)
                state = self.copy_state(
                    self.cpucells[name].get_state(), device=device
                )
                cell.load_state(state)
            else:
                cell = self.create_bead(
                    name, 'set_and_load')

            self.cells_online[device][name] = cell
        if not name in self.cell_mdevice:
            self.cell_mdevice[name] = device
        return cell
        
    def copy_state(self, state, device = 'cpu'):
        '''hard copies the state or checkpoint of a cell, makes non-trivial
            copies of tensors
            state:dict: cell's checkpoint
            device: str or torch.device
            ----return state_copy:dict
        '''
        max_recursion = 5
        def recursive_copy(state, state_copy, key = None, recur = 0):
            if recur + 1 > max_recursion:
                return
            if isinstance(state[key], (list,tuple,dict)):
                if isinstance(state[key], (dict, list)):
                    state_copy[key] = state[key].copy()
                else:
                    state_copy[key] = state[key]
                if isinstance(state[key], (tuple, list)):
                    iterable = range(len(state[key]))
                else:
                    iterable = state[key]
                for i in iterable:
                    recursive_copy(state[key], 
                        state_copy[key], i, recur+1)
            elif torch.is_tensor(state[key]):
                state_copy[key] = state[
                    key].detach().clone().to(device=device)
        state_copy = state.copy()
        for i in state:
            recursive_copy(state, state_copy, i, 0)
        return state_copy

    def pull_cells(self, device, one = False, criteria='random'):
        '''pull one cell to CPU, or what is given by the device
            device: str OR torch.device:
            one: bool: pull one cell or pull the amount required to get to the
                soft limit of the device
            criteria: str: random OR fitness
            ----return bool: True if at least one cell was pulled
        '''
        all_cell = len(self.cells[device]) + len(self.cells_online[device])
        dcells = self.cells[device]
        limit = self.acc_devices[device]['soft_limit']
        return self._pull_cells(dcells, all_cell, limit, one = one, 
            send_cpu=True, criteria=criteria)

    def pull_cells_cpu(self,):
        '''pull CPU cells to files, randomly
            ----return bool: True if at least one cell was pulled
        '''
        all_cell = len(self.cpucells)
        dcells = self.cpucells
        limit = self.devices['cpu']['soft_limit']
        return self._pull_cells(dcells, all_cell, limit, one = False, 
            send_cpu=False, criteria='fitness')

    def _pull_cells(self, 
            dcells, all_cell, limit, 
            one = False, send_cpu = False,
            criteria = 'random'):
        '''Inner function used to pull cells and check the criteria and limits
            dcells: dict: key, name of the cell, values, cells
            all_cell: int: all the cells to check
            limit: int: cells the device will have after pulling the rest
            one: bool: pull one cell or by the limit
            send_cpu: bool: send cell to cpu
            criteria: str: criteria to use, random OR fitness
            fitness => n(times the cell has been used)/loss(avg loss)
            ----return bool: True if at least one cell was pulled
        '''
        sub = all_cell - limit
        if sub > 0 and len(dcells) > 0:
            k = 1 if one else sub
            if criteria == 'random':
                keys = random.sample(dcells.keys(), k)
            elif criteria == 'fitness':
                all_cell_fitness = []
                for cell in dcells.values(): #choose the bottom worst
                    loss = cell.statistics['loss']
                    n = min(self.settings['n_threshold'], cell.statistics['n'])
                    fitness = (n/self.settings['n_threshold'])/(loss + 1e-5)
                    all_cell_fitness.append((fitness, cell))
                all_cell_fitness.sort(key=lambda x: x[0])
                cell_2del = all_cell_fitness[:k]
                keys = [x[1].name for x in cell_2del]
            for name in keys:
                state = dcells[name].get_state()
                if send_cpu:
                    cpu_state = self.copy_state(state)
                    self.cpucells[name] = OffCell(
                        name, self.bead_folder, self.settings
                    )
                    self.cpucells[name].load_state(cpu_state)
                else:
                    dcells[name].save_state()
                del(dcells[name])
            return k
        return False

    def update_states(self, ):
        '''Updates the states of cells in other devices
            GPU to CPU to HDD
            ----return bool: True if offline update(save states to HDD)
        '''
        self.update_states_online()
        self.cycles += 1
        if self.cycles % self.save_freq == self.save_freq - 1:
            print('Saving checkpoint at: ' + str(self.cycles))
            self.update_states_offline()
            return True
        self.update_tickings()
        return False
        
    def update_states_online(self): 
        '''transfers states from GPU to CPU, at the end of a cycle
        '''
        for device in self.acc_devices:
            self.cells[device].update(self.cells_online[device])
            self.cells_online[device] = {}
            self.pull_cells(device, one=False, criteria='fitness')

    def update_tickings(self): 
        '''Updates tickings, ticking is an oscilatory function
            of a number of cycles, modulo n of the number of cycles
        '''
        tick_cycles = self.settings[
            'internal_values_components']['tick_encoding']
        tickings = []
        for tick in tick_cycles:
            #tickings += list(
            #            utils.osct(self.cycles%tick, tick)
            #        ) #tuple -> list -> concat to list
            tickings.append( (self.cycles%tick)/tick ) 
        self.tickings = tickings

    def update_states_offline(self,): 
        '''CPU and GPU to files, every Nth or so, also terminals
        '''
        for device, device_cells in self.cells.items():
            for name, cell in device_cells.items():
                '''
                if name in self.cell_mdevice:
                    mdevice = self.cell_mdevice[name]
                    if mdevice != device and name in self.cells[mdevice]:
                        continue
                '''
                cell.save_state() #doesn't matter device, latest prevails
        for cpu_cell in self.cpucells.values():
            cpu_cell.save_state()
        for io_term in self.terminals.values():
            for term in io_term:
                term.cell.save_state()
                term.save_parameter_values()
        self.pull_cells_cpu()

    def check_mem_device(self, device):
        device_sett = self.devices[device]
        return utils.get_memory_used(device_sett)
    
    def get_mem_loss(self, device):
        mem_used = self.check_mem_device(device)
        x_n = (mem_used-self.mloss_t)**self.mloss_p
        return (x_n)/(1-x_n)

    def sync_cells(self):
        # TODO for more than one GPU
        # set weights of gpu cell to other devices
        # in split mode, not io_cells
        pass

class CellHierarchicalMemoryManagerSlave(CellHierarchicalMemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_main(is_slave= True)
        self.r_masks, self.r_matrix = self.get_routing_masks()
        for io in self.settings['terminal_io']:
            self.cell_terminals += self.terminals[io]
        self.all_cells_list = [
            *self.cells_list,
            *self.cell_terminals,
        ]
        #self.all_used = {x:False for x in self.devices.keys() if x!= 'CPU'}
        #self.access_statistics = {}
        self.end_device = {x:False for x in self.acc_devices}

class CellHierarchicalMemoryManagerMaster(CellHierarchicalMemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_main(is_slave= False)

class CellHierarchicalMemoryManagerOffline(CellHierarchicalMemoryManagerSlave):
    def __init__(self, *args, mode='create', **kwargs):
        '''Child of CellHierarchicalMemoryManager, deals with all memory functions
            that are done in offline mode, not training
            mode: str: what offline operation to do the pool of cells
        '''
        self.slave_terminals = True
        self.mode = mode
        if mode == 'create':
            init_behaviour = 'set_and_save'
        elif mode == 'reset':
            init_behaviour = 'reset_and_save'
        elif mode == 'expand':
            init_behaviour = 'set'
        elif mode == 'load':
            init_behaviour = 'set_and_load'
        elif mode == 'mutate':
            init_behaviour = 'set_and_load'
        else:
            print('Option now recognized')
            assert False
        super().__init__(*args, 
            init_behaviour= init_behaviour,**kwargs)
        if mode == 'mutate':
            self.mutate_bead_pool()
        
    def set_terminals(self):
        '''Sets the terminals with the offline modes and without special ones
        '''
        for io_terms in self.terminals.values():
            for terminal in io_terms:
                terminal.device = 'cpu'
                terminal.set_cell(
                    init_behaviour = self.init_behaviour
                )
                if self.mode in ['create', 'reset']:
                    terminal.create_parameters()
                    terminal.save_parameter_values()
                if self.mode == 'expand':
                    terminal.cell.model_init(
                        'expand_and_save'
                    )

    def mutate_bead_pool(self,): 
        '''done starting a sleep phase, once a week
            mutates the weights and bias of all cells, random mutations
            crossovers of cells with low fitness
        '''
        #----------------------GET FITNESS------------------------------
        all_cell_fitness = []
        mut_param = self.settings['mutate_param']
        for cell in self.cells['cpu'].values(): #choose the bottom worst
            i = int(cell.name,2)
            loss = cell.statistics['loss']
            n = min(self.settings['n_threshold'], cell.statistics['n'])
            fitness = loss*(n/self.settings['n_threshold'])
            all_cell_fitness.append((fitness, cell))
        all_cell_fitness.sort(key=lambda x: x[0])
        for n, (fitness, cell) in enumerate(all_cell_fitness):
            uname = cell.name
            if n > int(mut_param['bottom']*self.routing_dim**2) or cell.statistics['loss'] < 0.1:
                #reduce number of mutations if not in bottom
                cell.statistics['mutated'] = min(-10, cell.statistics['mutated'] - 1)
                cell.save_state()
                continue
            if random.random() < mut_param['reset'] or cell.statistics['mutated'] > 10:
                #if there's more than n mutations then reset the cell, or by a p very small
                #(mutation not improving the cell)
                cell = self.create_bead(uname, device = 'cpu', mode = 'reset_and_save')
                continue
            #----------------------GET WORST------------------------------
            cell.statistics['mutated'] += 1
            diff_loss_old = cell.statistics['diffuseness']
            diff_loss_all = [] #compare diffuseness
            for new_cell in self.cells['cpu'].values():
                if new_cell.name == cell.name:
                    continue
                diff_values_new = new_cell.statistics['diffuseness']
                diff_loss = torch.nn.functional.mse_loss(diff_loss_old, diff_values_new)
                diff_loss *= (new_cell.statistics['n']/self.settings['n_threshold'])
                diff_loss_all.append((diff_loss, new_cell))
            diff_loss_all.sort(key=lambda x: x[0])
            #----------------------GET GOOD CANDIDATE ------------------------------
            to_mutw = random.choice(
                diff_loss_all[:int(mut_param['take_best']*len(diff_loss_all))]
            )[1]
            checkpoint = cell.get_checkpoint()
            model_state = checkpoint['model_state_dict']
            model_state_new = to_mutw.get_checkpoint()['model_state_dict']
            if random.random() < mut_param['duplicate']:
                model_state = model_state_new
                model_state_new = model_state
            for mod_name, mod_param in model_state.items():
                mod_param_new = model_state_new[mod_name]
                #---------------------CROSS OVER OPERATIONS-----------------------------
                cross_p = random.random()
                cross_mutp = mut_param['crossover_p']
                cp = 0
                mutk = 'none'
                for kc, cpi in cross_mutp.items():
                    cp += cpi
                    if cross_p < cp:
                        mutk = kc
                        break
                if mutk == 'none':
                    pass
                else:
                    orig_size = mod_param.size()
                    if mutk == 'magnitude':
                        flat_mod = mod_param.view(-1)
                        flat_mod_new = mod_param_new.view(-1)
                        mag_sort = torch.argsort(flat_mod)
                        mag_sort_2 = torch.argsort(mag_sort)
                        #https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
                        mag_sort_new = flat_mod_new.sort()[0][mag_sort_2]
                        model_state[mod_name] = mag_sort_new.view(orig_size)
                    elif 'groups' in mutk:
                        flat_mod = mod_param.view(-1)
                        flat_mod_new = mod_param_new.view(-1)
                        p = 1/flat_mod.size()[0]**0.5
                        cuts = torch.rand_like(flat_mod) < p
                        cuts = torch.nonzero(cuts)
                        si = 0
                        for sf in cuts:
                            sf = sf[0]
                            if 'magnitude' in mutk:
                                mag_sort_2 = torch.argsort(
                                    torch.argsort(flat_mod[si:sf]))
                                to_set = flat_mod_new[si:sf].sort()[0][
                                    mag_sort_2]
                            else:
                                to_set = flat_mod_new[si:sf]
                            flat_mod[si:sf] = to_set
                            si = sf
                        flat_mod[si:] = flat_mod_new[si:]
                        model_state[mod_name] = flat_mod.view(orig_size)        
                    elif mutk == 'swap_layers':
                        model_state[mod_name] = mod_param_new
                            
                #----------------------POINT MUTATION---------------------------------
                nostop=True
                while(nostop):
                    param = None
                    for i in mod_param.size():
                        if param == None:
                            param = mod_param
                        rnd_idx = int(i*random.random())
                        if i == len(mod_param) - 1:
                            param[rnd_idx] = random.random()
                        else:
                            param = param[rnd_idx]
                    nostop = random.random() < mut_param['point']
            checkpoint['model_state_dict'] = model_state
            cell.load_state(checkpoint)
            cell.save_state()
            

class CellHierarchicalMemoryManagerCPU(CellHierarchicalMemoryManager):
    def update_states_online(self, *args, **kwargs):
        return
