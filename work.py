
import time
import os
import torch
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
from utils import dynamic_average, curiosity_value, dynamic_average_pyt
from cycle_graph import CycleGraph

class Cycle():
    '''
        Cycle = constructor, it runs the cycle graph, collects the output,
        gathers the output of all chains onto terminals, and takes the loss,
        and does the optmizer step, and cleans everything for the next cycle
    '''
    def __init__(self, 
            Hmemory, inspect, settings):
        self.Hmemory = Hmemory
        self.settings = settings
        self.inspect = inspect
        #self.big_loss_avg = big_loss_avg
        #self.rwdact = lambda x: torch.tanh(x)
        #sets all the loss lamndas
        self.len_sources = len(self.settings['all_sources']['Output'])
        self.len_types = len(self.settings['all_types'])
        self.l_cd = self.settings['loss_lambdas']['reward_critic']
        self.l_types = self.settings['loss_lambdas']['types']
        self.l_rwdval = self.settings['loss_lambdas']['reward_value']
        self.l_cur = self.settings['loss_lambdas']['curiosity']
        tlredfn = self.settings['terminal_loss_reduction_function']

        if tlredfn == 'softmax':
            fn = lambda l: torch.sum(torch.softmax(l, 0))
        elif tlredfn== 'normalize':
            fn = lambda l: torch.sum(F.normalize(l, dim=0)) #also mean could work
        elif tlredfn == 'sum':
            fn = lambda l: torch.sum(l)
        elif tlredfn == 'mean':
            fn = lambda l: torch.mean(l)
        self.tl_reduction_fn = fn
        self.cells = {}
        self.init_time = 0
        self.phase = None
        self.rewards = None

        self.out_device = self.Hmemory.dev_dist['terminals']['Output'][0]
        loss_keys_nd = ['big_loss', 'big_loss_change', 
            'loss_change', 'loss_crit_real']
        self.losses_dict = {}
        for key in loss_keys_nd:
            ls = torch.tensor(0.0, device=self.out_device)
            ls.requires_grad = False
            self.losses_dict[key] = ls

    def train(self, to_back, graph):
        '''
            Main training fuction of Universal Slave
            1) uses the scalar losses defined in __init__
            takes the time so it compares with the time in __init__
            and defines some constants
            2) loops through all terminals taking first the critic loss, this loss and the difference
            is used for the norm loss in the real loss and then the virtual loss.
            A penalty is incurred if the critic terminal is not used
            3) backpropagates all the cells' losses that are saved in to_back
            it could be a dead cell, a predictor or whatever.

            to_back: list: each element with a tuple of the values to backpropagate,
            the choosen value(if categorical), and the type of backprop
            ---return:(
                big_loss: tensor(1,1): average of all terminals' losses
                big_loss_change: tensor(1,1): average of all terminals' change in loss
            )
        '''
        out_device = self.out_device
        big_loss = self.losses_dict['big_loss'] #used in loss due to death cells
        big_loss_change = self.losses_dict['big_loss_change'] #used to inhibit deepness
        now_t = time.time()
        rewards = self.rewards #{'type': float}
        ended_term = []
        all_param_losses = []
        #It has to be trained first the type, then the source
        for sidx, src in enumerate(self.settings['all_sources']['Output']):
            critic_to = 'Critic' if self.phase['wake'] else 'CriticDream'
            critic_terminal = self.Hmemory.get_terminal(
                'Output', (critic_to, src)
            )
            #-----------------------------CRITIC--------------------
            if not critic_terminal:
                raise ValueError(
                    critic_to + src, ' doesn\'t have critic terminal ',
                    ' ALL terminal sockets (systems) must have critic as guide for the rest',
                    ' of terminal of the same source'
                )
            past_param_closs = critic_terminal.extra_parameters[
                'last_param_loss']
            if len(critic_terminal.traversed) > 0:
                seq_inter = critic_terminal.forward_fn(
                    self.Hmemory, self.phase
                ) 
                #param1: from input terminal in last cycle
                #param2: from input terminal in actual cycle
                param_closs = critic_terminal.take_base_loss(
                    [], seq_inter)
                all_param_losses += [lss * self.l_types[ 
                    'Critic'] for lss in param_closs.values()]
                ended_term.append(critic_terminal)
                critic_terminal.set_parameter_values(
                    param_closs, 'last_param_loss', detach_copy = True,
                    no_grad = True
                )
            else:
                param_closs = critic_terminal.extra_parameters[
                    'last_param_loss']

            #-----------------------------REAL/VIRTUAL----------------------------
            ac_types = [
                'Real' if self.phase['wake'] else 'Dream',
                'Virtual' if self.phase['wake'] else 'VirtualDream',
            ]
    
            for i, at in enumerate(ac_types):
                out_terminal = self.Hmemory.get_terminal(
                    'Output', (at, src)
                )
                if not out_terminal:
                    continue
                if out_terminal.traversed > 0:
                    prev_param = out_terminal.forward_fn(
                        self.Hmemory, self.phase)
                    #Critic indirect loss
                    param_rloss = out_terminal.take_reward_loss(
                        prev_param, param_closs, past_param_closs,
                        self.l_cd
                    )
                    #diff_lrwd = 1 + out_terminal.diffuseness_loss()/2
                    ld = self.l_types[at]
                    all_param_losses += [lss*ld for lss in param_rloss.values()]
                    ended_term.append(out_terminal)
                else:
                    pass
                    #not_ended_term.append(out_terminal)

        #------------------- GET BIG LOSS ----------------

        all_losses = [loss.to(
            self.out_device) for loss in all_param_losses]
        term_loss = torch.cat(all_param_losses)
        term_loss = torch.mean(term_loss.detach().clone())
        for tbck in to_back:
            sam_action = tbck[0]
            values_clone = sam_action[1]
            values = values_clone
            if tbck[1] == 'curiosity':
                #extra_values = sam_action[0]
                cvalues = curiosity_value(sam_action[0])
                loss_fn = torch.nn.CosineSimilarity(0)
                loss = loss_fn(values, cvalues)*self.l_cur
            elif tbck[1] == 'reward':
                loss = torch.nn.functional.mse_loss(
                    sam_action[0], values
                )*self.l_rwdval
            all_losses += loss.to(device = out_device)
        
        cell_to_train = self.Hmemory.all_cells_list
        cell_fractions = torch.zeros(len(
            self.settings['terminal_io']), 
            self.settings['diffuseness_dim'],
            self.settings['n_all_cells'],
            device=self.out_device) #io, term, cells
        if rewards:
            output_fractions = torch.zeros(
                self.settings['diffuseness_dim'],
                self.settings['seq_len'],
                self.settings['n_all_cells'],
                device=self.out_device
            ) #term(output), seq, cells + 2*term(output)
        else:
            output_fractions = None

        for cell in self.Hmemory.cell_terminals: # iterate terminal cells for cell countings
            cell_f = cell.get_cell_count_fraction()
            term_index = cell.terminal.index #not diff index
            io_index = cell.io_index
            cell_f = cell_f.to(self.out_device)
            cell_fractions[io_index][term_index] += cell_f
            if rewards and cell.terminal.category_dict['io'] == 'Output':
                cell_fhist = cell.model.activation_hist.get_buffer(
                    'hist')[0]
                output_fractions[term_index] += cell_fhist

                # seq, cells + 2*term
        cell_fractions_perm = cell_fractions.permute(1, 0, 2) # cells, io, term
        if rewards:
            for cell in cell_to_train:
                all_losses += [cell.cell_reward_train(rewards, output_fractions)]
        all_losses = self.tl_reduction_fn(all_losses)
        all_losses.backward()
        all_losses = all_losses.item()
        for cell in cell_to_train:
            self.train_cell(cell, now_t, all_losses, cell_fractions_perm)
        return all_losses

    def train_cell(self, cell, now_t, loss, cell_fractions):
        ''' cell: cell, now_t: float, loss: loss
            cell_fractions_perm: cell fractions for diffuseness 
        '''
        if cell.train and cell.traversed > 0:
            cell.set_lr(self.phase['phase_change_lr'])
            #cell.cell_reward_loss()
            if not self.phase['wake']:
                torch.nn.utils.clip_grad_value_(cell.model.parameters(), 0.5)
            cell.set_weight_decay()
            cell.optimizer.step() #only parameters() change
            cell.update_statistics(now_t, loss, cell_fractions)
            cell.cycle_notraversed = 0
        else:
            cell.cycle_notraversed += 1

        cell.clean_next() #set a matrix of zeros as the value to shift in hist 

    def clean_run(self,):
        '''
            Cleans the terminals in memory
        '''
        
        terminals = [*self.Hmemory.terminals['Output'], 
            *self.Hmemory.unique_terminals.values()]
        for term in terminals:
            term.clean_next()

    def collect_rewards(self,):
        '''
            collects rewards updated in file from another terminal,
            then it blanks the file
            rewards: tensor: current rewards
            rewards_fraction: tensor: fraction of the reward valid
                by update time
            all_sources: dict: key as source: value as index of diff_value
            types_available: dict: key as terminals: value as types available
            device: torch.device
        '''
        rewards = torch.zeros(self.settings[
            'diffuseness_dim'], device=self.out_device)
        rf = open('./reward.txt','w+')
        readlines = rf.read().split(';')
        rf.write('')
        rf.close()
        #array 2 elements, hard and soft
        if len(readlines) > 0 and len(readlines[0])>0:
            for line in readlines:
                source, rwd = line.split(':') #source and reward
                real_terminal = self.Hmemory.get_terminal(
                    'Output', ('Real', source)
                )
                indx = real_terminal.index
                if source in self.settings['all_sources']:
                    rewards[indx] = rwd
                else:
                    raise ValueError('source '+ source +' not in rewards')
        else:
            rewards = None
        return rewards

    def set_lock(self,):
        '''sets and removes lock for master/slave'''
        slockfp = './external/slave_lock'
        mlockfp = './external/master_lock'
        if os.path.exists(mlockfp):
            os.remove(mlockfp)
        f = open(slockfp, 'x')
        f.write('')
        f.close()

    def wait_for_master(self,):
        '''waits for master to collect the media
        '''
        while(os.path.exists('./external/slave_lock')):
            time.sleep(0.001)

    def run(self, phase):
        '''
            Main method, creates the CycleGraph and runs it
            ---return:(
                big_loss: tensor(1,1): average of all terminals' losses
                bis_loss_change: tensor(1,1): average of all terminals' change in loss
            )
        '''
        self.cells = {}
        self.init_time = time.time()
        self.phase = phase
        to_back = []
        if self.phase['wake']:
            self.rewards = self.collect_rewards()
        else:
            self.rewards = None
        self.wait_for_master()
        for in_terminal in self.Hmemory.main_input_terminals:
            #traverses only the values that are shared later
            parameters_ls = in_terminal.forward_fn(
                self.Hmemory, self.phase)
            values = in_terminal.cell.model.forward_term(
                parameters_ls
            )
            in_terminal.shared_intervalues = values

        graph = CycleGraph(
            Hmemory = self.Hmemory, phase = self.phase,
            settings = self.settings, store_connections = self.inspect
        )
        graph.traverse()
        to_back += graph.to_back
        big_loss, big_loss_change = self.train(to_back, graph)
        self.clean_run()
        self.set_lock()
        #update cell statistics
        return big_loss, big_loss_change