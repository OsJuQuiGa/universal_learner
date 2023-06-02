import torch
import time
import datetime
import utils


class CycleGraph(): 
    '''
        Constructor for object that traverses each terminal and
        branches more chains to keep traversing the cells, eventually
        leading to the output terminals or dying, by not chosing to do so
        or by reaching their limits
        in_terminal: terminal_obj:
        store_connections: bool: stores the connections in string form
            for displaying at the end of cycle
        Hmemory: Hmemory_obj:
        phase: dict:
        settings: dict:
    '''
    def __init__(self,Hmemory=None, phase = {}, settings = {}, 
            store_connections = False):

        self.settings = settings
        self.in_terminals = Hmemory.terminals['Input']
        self.out_terminals = Hmemory.terminals['Output']
        self.Hmemory = Hmemory
        self.store_connections = store_connections
        self.in_graph = {}
        self.out_graph = {}

        self.hard_limit_routings = self.settings['cell_limits']['routing']['max']
        self.tickings = Hmemory.tickings
        self.phase = phase
        self.init_time = time.time()
        self.routings = 0

        self.to_back = []
        self.online_cells = []
        self.offline_cells = {}
        cell_dev = Hmemory.acc_devices['cells'][0]
        self.penalty = torch.zeros(self.settings['n_all_cells'] + 1, device=cell_dev) 
        self.out_diffuseness = []
        for cell in Hmemory.all_cells_list:
            div = self.settings['no_traverse_penalty_div'][cell.type]
            self.penalty[cell.diff_index] = cell.cycle_notraversed/div
            self.out_diffuseness.append(
                cell.statistics['diffuseness'][1].to(cell_dev)
            )
        rand_diff = torch.rand_like(
            self.out_diffuseness[-1])*(1/settings['diffuseness_dim'])
        self.out_diffuseness.append(rand_diff)
        self.out_diffuseness = torch.stack(self.out_diffuseness) 
        #(cellterm),term -> cellterm, term

    def traverse(self, ):
        ''' Inits chain and traverses the input terminal
        '''
        for in_terminal in self.in_terminals:
            self.traverse_input_terminal(in_terminal)
        self.iterate_cells()
        for out_terminal in self.out_terminals:
            self.traverse_output_terminal(out_terminal)

    def traverse_input_terminal(self, in_terminal):
        '''Traverses the input terminal of the chain
            ----return where_list_act: list:
        '''
        if in_terminal.shared_intervalues is None:
            in_terminal.forward_fn(
                self.Hmemory, self.phase
            )
            
        cat_dict = in_terminal.shared_category_dict
        shared_terminal = self.Hmemory.get_terminal('Input',
                (cat_dict['type'], in_terminal.cat_dict['source']))

        values = shared_terminal.shared_intervalues.copy()
        cell = in_terminal.cell
        cell_counts = torch.zeros_like(cell.cell_counts)
        cell_back_routing = [cell]
        cell.set_values_cell(0, values, 
            self.settings['n_cells'] + in_terminal.index,
            cell_counts, cell_back_routing, 0,
            maturity = in_terminal.cell.statistics['maturity'],
        )
        self.traverse_cell(cell)
        self.online_cells.append(cell)

    def traverse_output_terminal(self, out_terminal):
        '''Traverses the output terminal of the chain, if reached
            traverses the cell of the terminal, gets the parameters in list form
            so it can be expressed by param_names defined inside the terminal
            out_terminal: terminal_obj:
        '''
        if out_terminal.traversed > 0:
            cell = out_terminal.cell
            self.short_traverse_cell(cell)
            out_terminal.expression_end_cycle(cell.cell_values)
            dindx = self.settings['n_cells'] + self.settings[
                'diffuseness_dim'] + out_terminal.index
            cell.cell_counts[dindx] = cell.n_input
            
    def base_traverse_cell(self, cell):
        ''' Base for all traversions'''
        if cell.train and cell.traversed < 1:
            cell.optimizer.zero_grad()
        cell.traversed += 1 #same as deepness
        now_t = time.time()
        return now_t

    def short_traverse_cell(self, cell):
        ''' Only for Output terminal
        '''
        now_t = self.base_traverse_cell(cell)
        cell.get_cell_entries() #called in outputterminal method
        in_values = cell.cell_values
        pre_values = cell.pre_traverse(in_values, now_t)
        main_values = cell.short_inner_traverse(pre_values, now_t)
        cell.cell_values = cell.extra_traverse(main_values, in_values)

    def traverse_cell(self, cell):
        ''' For output terminal and cells'''
        now_t = self.base_traverse_cell(cell)
        cell.get_cell_entries()
        in_values = cell.cell_values
        pre_values = cell.pre_traverse(in_values, now_t)
        r_mask = self.Hmemory.r_masks[cell.device] # phase mask, #{key}, conn
        main_values = cell.inner_traverse(
            pre_values, now_t, r_mask)
        out_values = cell.extra_traverse(main_values, in_values)
        cell.cell_values = out_values
        #cell.update_diffuseness

    def iterate_cells(self,):
        ''' Infinite loop checks for online cells, also accumulates the values 
            to backpropagate of the chains
        '''
        end_cond = False
        while(len(self.online_cells) > 0 and not end_cond):
            cells_iterator = self.online_cells
            still_online = {}
            for cell in cells_iterator: #routes, and accumulates in cells
                self.to_back += cell.to_back
                cell.to_back = []
                cells_route = self.do_where_actions(cell) #these are pool cells (no output terminal)
                self.routings += len(cells_route)
                for cell_ol in cells_route:
                    still_online[cell_ol.name] = cell_ol
            for cell in still_online.values(): #traverses this accumulation of routes
                self.traverse_cell(cell)
            self.online_cells = still_online.values()
            self.offline_cells.update(still_online)
            end_cond = self.routings > self.hard_limit_routings

        return 'FINISHED'
        
    def do_where_actions(self, cell):
        ''' makes the actions that where sampled from the
            cells, either keep traversing the cell pool or
            traversing a terminal
            chain: chain_obj: current chain
            where_list_act: list: samples and values tensor of
                where to traverse
        '''
        cells_to_iter = []
        where_act = cell.where_act
        cell.where_act = []
        maturity = cell.statistics['maturity']
        cell_values = cell.cell_values #B(*N), L

        for widx in range(len(where_act['connections'])):
            conn = where_act['connections'][widx]
            to_route = where_act['to_route_values'][widx]
            connindx = where_act['indeces'][widx]

            ridx = conn['receptor'][widx]
            conntype = conn['cell_type']
            if conntype == 'Kill':
                continue
            elif conntype == 'Cell': #routing
                code_bin = conn['where'] # 0001110
                cell_rout = self.Hmemory.get_cell(code_bin)
                cells_to_iter.append(cells_to_iter)
                cr_indx = cell_rout.diff_index
                for term_cell in cell.cell_back_routing:
                    term_cell.cell_counts[cr_indx] += 1 #in terminal cell

            elif conntype == 'Terminal': #outputing
                #print(where_act[1][0])
                source = conn['source']
                ttype = conn['type']
                terminal = self.Hmemory.get_terminal(
                    'Output',ttype,source)
                terminal.init_time = self.init_time
                terminal.traversed += 1
                cell_rout = terminal.cell
            
            cycle_value = cell_values[widx]*to_route.unsqueeze(0)
            cell_rout.set_values_cell(
                ridx, cycle_value, cell.index,
                cell.cell_counts, cell.cell_back_routing, cell.cycle_reward,
                maturity = maturity,
            )
        return cells_to_iter

