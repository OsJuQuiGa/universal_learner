from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import copy
from utils import dynamic_average_pyt

from generators.extra_models.utils import Null
import utils
import generators.models as Models


class WhereModel(nn.Module):
    def __init__(self, model, routings, connections):
        super().__init__()
        self.routings = routings
        self.connections = connections
        self.model = model
        
    def penalty_toback(self, where_output, penalized_connections, count_connections):
        ''' 
            penalizes connections that are not being used in the cycle, also
            connections that are repeated too much
            penalized connections are updated at the end of the cycle
            where_output: B, conn
            penalized_connections: conn
        '''
        penal_count = 1/(torch.log(count_connections + 1) + 1)
        mod_where_output = where_output.detach().copy()
        modval, values = None, None
        with torch.no_grad():
            mod_where_output[:, :] = mod_where_output[:, :] * \
                penalized_connections * penal_count 

        modval = F.log_softmax(mod_where_output)
        values = F.log_softmax(where_output.copy())
        return (modval, values)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs) # B, routing

class CellReward(nn.Module):
    def __init__(self, main_dim, n_context = 16, n_terminals= 1, use_q = False, 
            use_q_after_n = True, n = 100000):
        super().__init__()
        self.register_buffer('context', torch.rand(n_context, main_dim)) #action context
        self.n = n
        #for inmmatured this is active
        self.register_buffer('context_reward', torch.zeros(n_context, n_terminals))
        self.value_net = nn.Sequential(
            nn.Linear(main_dim, main_dim), 
            nn.GELU(),
            nn.Linear(main_dim, n_terminals)
        )
        self.use_q = use_q
        self.n_context = n_context
        self.use_q_after_n = use_q_after_n
        if use_q: #q value, when the network has matured, then this is active
            self.q_net = nn.Sequential(
                nn.Linear(main_dim, main_dim), 
                nn.GELU(),
                nn.Linear(main_dim, n_context)
            )

    def store_context(self, out, sim_mm, context):
        '''out: B, L; sim_mm: B, ctx; context: ctx, L
        '''
        ctxmax = torch.argmax(sim_mm, dim=1) #B(int)
        for i, ctxi in enumerate(ctxmax): #Batch reduction
            ctxilat = context[ctxi] #L
            context[i] = dynamic_average_pyt(ctxilat, out[i], self.n)
        self.register_buffer('context', context)

    def forward(self, out, n):
        '''out: B, L'''
        #value
        value = self.value_net(out) # B, term
        with torch.no_grad():
            context = self.get_buffer('context') # ctx, L
            context_reward = self.get_buffer('context_reward') # ctx, term
            sim_mm = torch.einsum('bl, cl -> bc', out, context) # B, ctx
            sim = torch.softmax(sim_mm, dim=1) # B, ctx
            value_reward = torch.einsum('bc, ct -> bt', sim, context_reward) # B, term
            self.store_context(out, sim_mm, context)
        to_train = [(value, value_reward)] #B, term X2

        if self.use_q:
            with torch.no_grad():
                value_reward_mm = torch.einsum(
                    'bt, ct -> bc', value.detach(), context_reward) #B, ctx
            q_values = self.q_net(value) #B, ctx
            if self.use_q_after_n and n > self.n:
                qlat = self.forward_q(q_values, context, out, value) #B, L
            to_train.append((q_values, value_reward_mm))
        else:
            qlat = out #B, L
        return to_train, qlat

    def forward_q(self, q_values, context):
        '''out B, L; value B, ctx''' # ctx, L
        q_values = torch.softmax(q_values, dim=1) #B, ctx
        qlat = torch.einsum('bc, cl -> bl', q_values, context)#/self.n_context #B, lat
        return qlat
    @torch.no_grad()
    def self_curiosity(self, output_seq):
        '''output_seq: seq, lat
        '''
        seq_dim = output_seq.size()[0]
        curiosity = torch.einsum('sq , qs -> ss', output_seq, output_seq) # seq, seq
        diag_div = torch.outer(torch.diag(curiosity), torch.ones(seq_dim)) # seq, seq
        curiosity = curiosity/diag_div
        seq_curiosity = torch.mean(
            torch.abs(curiosity - 1), dim=1) + 1 # seq
        return seq_curiosity # seq
    @torch.no_grad()
    def train(self, output_sec, drwd, n):
        '''output_sec: seq, lat
            drwd: seq, term; discounted reward from diffuseness

        '''
        context = self.get_buffer('context') # ctx, L
        context_reward = self.get_buffer('context_reward') # ctx, term
        seq_len = output_sec.size()[0]
        seq_disc = torch.linspace(0.1, 1, seq_len, device=output_sec.device) #seq
        seq_disc[-1] = 0.5
        seq_curiosity = self.self_curiosity(output_sec)
        seq_disc = seq_curiosity * seq_disc
        seq_rwd = seq_disc * drwd # seq, term
        seq_sim_mm = torch.einsum('sl, cl -> sc', output_sec, context) # seq, ctx
        seq_sim = torch.softmax(seq_sim_mm, dim=1) # seq, ctx
        ctx_term_rwd = torch.einsum('sc, st -> ct', seq_sim, seq_rwd) # ctx, term
        for rwd in ctx_term_rwd:
            context_reward = dynamic_average_pyt(
                context_reward, rwd, n)
        self.register_buffer('context_reward', context_reward)
        #return ctx_term, self.train_q(output_sec)

    def _train_q(self, output_sec):
        ''' output_sec: seq, b, lat
        '''
        if not self.use_value_q:
            q_train = self.q_net(output_sec) #seq, b, ctx
        else:
            q_train = self.value_net(output_sec) #seq, b, ctx
        return q_train

class CellFiltering(nn.Module):
    ''' Receptor confirms the values that are given by other cells,
        if values don't pass this adamask then these are lowered and the good
        values are increased.
        main_dim: features
        n_context: number of contexts to get close to
        n_segments: number of filters
        gather_mode:
            mean: it averages the receptor dimension
            batch: it makes batches of the number of inputs
            none: it sends receptors as batches, batch dim = batch * recep
    '''
    def __init__(self, main_dim, n_context = 4, n_segments = 4,
            n_reduction = 'mean'):
        super().__init__()
        self.main_dim = main_dim
        self.ctx_mod = nn.Parameter(
            torch.rand(n_segments, main_dim)
        ) 
        ctx_buffer = torch.rand(n_context, main_dim)
        self.receptor = nn.Sequential(
            nn.Linear(main_dim, main_dim),
            nn.GELU()
        )
        self.n_reduction = n_reduction
        if n_reduction is None:
            self.reductor = lambda x, dim: x
        else:
            self.reductor = getattr(torch, n_reduction)
        self.register_buffer('context', ctx_buffer)

    def get_update_context(self, x):
        '''x: B, L'''
        B = x.size()[0]
        x_n = x.detach().unsqueeze(1) #B, 1,  L
        ctx_buffer = self.get_buffer('context') # 1, nctx, L
        sim = F.cosine_similarity(
            x_n, ctx_buffer, dim=3
        ) # B, nctx
        argm = torch.argmax(sim, dim=2) # B
        context = [] #(B), L
        for b in range(B):
            argm_b = argm[b] #recep
            cls_ctx = ctx_buffer[0, argm_b] # L
            context.append(cls_ctx.clone())
            new_ctx = dynamic_average_pyt(cls_ctx, ctx_buffer[0, argm_b], 50000) # L
            ctx_buffer[0, argm_b] = new_ctx
        context = torch.stack(context, dim=0) #B, L
        self.register_buffer('context', ctx_buffer)
        return context

    def get_segments(self, context):
        ''' context: #B, L (argmax context)
        '''
        weights = self.ctx_mod #nseg, L
        seg_act = torch.einsum('sl, bl -> bs', weights, context) #B, nseg
        return seg_act

    def get_activation(self, segments):
        '''segments: #B, nseg'''
        max_seg, _ = torch.max(segments, dim=1,) #B
        return torch.sigmoid(max_seg).unsqueeze(dim=1) #B, 1

    def forward(self, x):
        '''the same linear model for all receptors, outputs B*N because
            it has to return these values to a transformer
            x: N, B, L,'''
        xs = x.size()
        x = torch.reshape(x, (xs[0]*xs[1], xs[2])) #B*N, L
        context = self.get_update_context(x) # B*N, L
        segments = self.get_segments(context) # B*N, nseg
        activation = self.get_activation(segments) # B*N, 1
        x_out = self.receptor(x) * activation # B*N, L
        if not self.n_reduction is None:
            x_out = torch.reshape(x_out, xs) #N, B, L
            x_out = self.reductor(x_out, dim=0) #B, L
        return x_out  #B, L or B*N, L without reduction

class CellModel(nn.Module):
    '''Base cell model for all cells, it has the

        ModelClass: torch.nn.Module: class of the main model
        Model_settings: dict: arguments passed to the model class
        settings: dict: global settings
        device: str or torch.device id of the pytorch device
        n_cells: int: subdivision of the values of the main model, this
        tackles the thin and deep issue for models with high capacity(terminals with images)
        use_subactions: list: subactions to include
        use_memory: bool: uses memory slots
        use_filtering: bool: uses a filter that predicts the 
        main values passes to cell with the internal values
        use_metatraining: TODO
        sample: dict override parameters of model settings
    '''
    def __init__(self,
            ModelClass = None,
            model_settings = {},
            settings = {},
            device = 'cpu',
            use_actions = ['where'],
            use_memory = False,
            use_metatraining = False, #not used
            cell_limits = {}, #default, and unique cell limits
            n_segments = 0,
            gather_receptor = 'mean',
            n_reward_contexts = 0,
            actions_settings_override = {}
        ):
        super().__init__()
        model_name = ModelClass.__name__
        #print(model_name)
        main_model = ModelClass(**model_settings).to(device)
        main_out_dim = main_model.get_out_dim()
        self.modules_to_clean = []
        if 'Hist' in model_name:
            self.modules_to_clean.append(main_model)
        self.main_model = main_model
        #mem_sett = settings['subactions']['memory']
        self.main_out_dim = main_out_dim
        self.norm_layer = nn.LayerNorm(main_out_dim)
        self.device = device
        self.memory_model = lambda x, *a, **k: x
        #self.routwouthead = self.settings['subactions']
        self.use_memory = use_memory
        self.use_actions = use_actions
        self.filtering_model = lambda x: x
        self.reward_model = lambda x: x
        self.relative_model = None
        act_sett = copy.deepcopy(settings['actions'])
        utils.update_dict(act_sett, actions_settings_override)
        self.msg_dim = settings['msg_dim']
        self.batch_size = 1
        assert self.main_out_dim % self.msg_dim == 0
        self.n_receptors = self.main_out_dim//self.msg_dim
        self.n_routings = 1
        self.n_connections = 1
        self.n_reward_contexts = n_reward_contexts 

        self.settings = settings
        self.cell_limits = cell_limits

        if self.n_receptors > 1:
            if gather_receptor == 'mean':
                self.receptor = self._mean_receptor
            elif gather_receptor == 'batch':
                self.receptor = self._batch_receptor
        else:
            if gather_receptor == 'mean':
                self.receptor = self._mean_one_receptor
            elif gather_receptor == 'batch':
                self.receptor = self._batch_one_receptor

        if use_memory:
            mem_sett = act_sett['memory']
            mem_mname = mem_sett['model']
            class_gen = getattr(Models, mem_mname)
            self.memory_model = class_gen(
                lat = main_out_dim, **mem_sett['model_settings'][mem_mname]
            )
        if n_segments > 0:
            self.filtering_model = CellFiltering(
                main_out_dim, n_segments, n_segments)

        if n_reward_contexts > 0:
            self.reward_model = CellReward(
                main_out_dim, n_context= n_reward_contexts,
                use_q=False, n=10000
            )
        for act in use_actions:
            act_settings = self.settings['actions'][act]
            act_model = self.get_head_model(
                main_out_dim, act_settings, 'first')
            if act == 'where':
                self.n_routings = act_sett[act]['routings']
                self.n_connections = act_sett[act]['connections']
                act_model = WhereModel(
                    act_model, act_sett[act]['routings'], 
                    act_sett[act]['connections'])
            setattr(self, act + '_model', act_model)

    def _mean_receptor_alt(self, cell_values):
        '''cell_values: (recep), (N), B, L(msg)
            for OutputTerminal, simplified input
        '''
        recep = [] #(recep), N, B, L
        for r in cell_values: #r: (N), B, L
            if len(r) == 0:
                recep.append(
                    torch.zeros(
                        (self.batch_size, self.msg_dim),
                        device=self.device
                    ) # 1, B, L
                )
            else:
                recep.append(
                    torch.mean(
                        torch.stack(r), dim=0, # N, B, L
                        keepdim=1 
                    ) # 1, B, L
                )
        out = torch.cat(recep, dim = 2) #1, B, Lmain
        return out
    def _mean_receptor(self, cell_values):
        '''cell_values: (recep), (N), B, L(msg)
            for OutputTerminal, simplified input
        '''
        x = self._batch_receptor(cell_values) #N, B, L
        return torch.mean(x, dim= 0, keepdim=1) #1, B, L

    def _mean_one_receptor(self, cell_values):
        '''cell_values: (1), (N), B, L
            For Cells, simplified input
        '''
        out = torch.stack(cell_values[0], dim=0) #N, B, L
        return torch.mean(out, dim=0, keepdim=True) #1, B, L

    def _batch_receptor(self, cell_values):
        '''cell_values: (recep), (N), B, L{msg}
            For Cells with msgdim smaller than main dim
        '''
        max_n = 0
        for r in cell_values:
            max_n = max(len(r), max_n)

        recep = [] #(recep), N, B, L
        for r in cell_values: #r: (N), B, L
            bdif = max_n - len(r)
            if bdif > 0:
                r.append(
                    torch.zeros(
                        (bdif, self.batch_size, self.msg_dim),
                        device=self.device), 
                )
            recep.append(
                torch.stack(r) # N, B, L
            )
        out = torch.cat(recep, dim = 2) #N, B, Lmain
        return out

    def _batch_one_receptor(self, cell_values):
        '''cell_values: (1), (N), B, L
            For Cells, and InputTerminals
        '''
        out = torch.stack(cell_values[0], dim=0) # N, B, L
        return out 

    def get_head_model(self, main_dim, act_settings, linker='none', reverse=False):
        ''' 
        gets the head model, with the intermediate main model of the head and the smaller models
         that come afterwards

         main_dim: int: dimension output of the cell's main model
         sact_settings: dict: settings for subaction, if settings are not
         given this sact_settings overtake
         ---- return nn.ModuleList: 
        '''
        if isinstance(act_settings['model'], str):
            model_name = act_settings['model']
            model_class = getattr(Models, model_name)
            model_settings = act_settings['model_settings'][model_name]
        else:
            model_name = act_settings['model'].__name__
            model_class = act_settings['model']
            model_settings = act_settings['model_settings']
        
        model = []
        if 'dim' in act_settings:
            features = act_settings['dim']
        else:
            features = main_dim
        if reverse:
            tmp = main_dim
            main_dim = features
            features = tmp

        if (features != main_dim and linker == 'first'):
            model += [('linker', Models.MLP(layers=2, 
                in_dim=main_dim, out_dim=features))]
        if model_name == 'HistTransformer':
            model += [('attntransformer', model_class(features,**model_settings))]
            self.modules_to_clean.append(model[-1][1])
        elif model_name == 'MemHistTransformer':
            model += [('memtransformer', model_class(features,**model_settings))]
            self.modules_to_clean.append(model[-1][1])
        elif model_name == 'MLP':
            model += [('mlp',Models.MLP(
                in_dim=features, out_dim=features,**model_settings))]
        else:
            model += [(
                model_name.lower(), 
                model_class(features,**model_settings))]

        if (features != main_dim and linker == 'last'):
            model += [('linker', Models.MLP(layers=2, 
                in_dim=main_dim, out_dim=features))]
        model = nn.Sequential(OrderedDict(model))
        return model

    def clean_next(self,):
        ''' cleans next model, specially models that keep the past and
            has to be averaged
        '''
        for model in self.modules_to_clean:
            model.next_cyle()

    def forward(self, values):
        main_values = self.main_model(values)
        return main_values

        
class CellHead(nn.Module):
    ''' class that contains the intermediate main head model, and the actions of the head.

     model: nn.Module the main intermediate model
     heads: list: list of nn.Modules, actions of the subaction
    '''
    def __init__(self, model = None, heads=None, slices = None, is_output=True,
            pn_used= None, pn_coupled = None):
        super().__init__()
        self.pn_used = pn_used
        self.pn_coupled = pn_coupled
        self.model = model
        self.heads = heads
        self.is_output = is_output
        if slices or len(slices) > 0:
            self.slices = slices
        else:
            self.slices = [slice(None,None) for i in heads]
        if is_output:
            self.forward = self._forward_output
        else:
            self.forward = self._forward_input

    def _forward_model(self, values):
        values = self.model(values)
        return values

    def _forward_head_input(self, values):
        head_values = [None]*len(self.pn_used)
        for i, head in enumerate(self.heads):
            pn = self.pn_used[i]
            kwargs = {} #safe
            if pn in self.pn_coupled:
                if self.pn_coupled[pn] is None:
                    continue
                else:
                    for pnc in self.pn_coupled[pn]:
                        idx = self.pn_used.index(pnc)
                        kwargs[pnc] = values[idx]
            hval = head(values[i],**kwargs)
                #kwargs argument is safe if it is a empty dict
            head_values.append(hval)
        return head_values

    def _forward_head_output(self, values):
        head_values = [None]*len(self.pn_used)
        for i, head in enumerate(self.heads):
            pn = self.pn_used[i]
            cpled = pn in self.pn_coupled
            if cpled and self.pn_coupled[pn] is None:
                continue
            hval = head(values[self.slices[i]])
            head_values[i] = hval
            if cpled:
                for pnc in self.pn_coupled[pn]:
                    idx = self.pn_used.index(pnc)
                    head_values[idx] = hval[1][pnc] # coupled models
                    #are expected to return values, {key:values}
        return head_values

    def _forward_output(self, values):
        out = self._forward_model(values)
        out = self._forward_head_output(out)
        return out
    
    def _forward_input(self, values):
        out = self._forward_head_output(values)
        out = torch.cat(out, dim = 1)
        out = self._forward_model(out)
        return out

class TerminalCellModel(CellModel):
    '''Cell model of terminals'''
    def __init__(self, terminal, **kwargs):
        '''it uses the parameter names list to construct the
            subactions, and the parameters defined inside each
            terminal and parameter settings, actions are the
            parameters here
        '''
        super().__init__(**kwargs)
        self.pnb_used = []
        param_names = terminal.param_names
        param_settings = terminal.parameters_settings
        self.activation_hist = Models.Hist(1, 0, 
            self.settings['seq_len'], self.settings['n_cells'],
            self.device, use_time_encoding= True, subdivisions=None)
        self.cat = terminal.category_dict
        self.parameters_couplings = {}
        self.model_heads = nn.ModuleList()
        is_output = self.cat['io'] == 'Output'
        self.is_main_term = is_output or terminal.shared_category == terminal.category
        if self.is_main_term:  #the other terminals that are shared with this
            #one uses the values of these modules
            internn = 0
            dft_subact = terminal.subaction_settings
            slices = []
            sindx = 0
            for _, pn in enumerate(param_names):
                psett = param_settings[pn]
                main_dim = psett['latent_dim']
                if psett['model'] is None: # or not self.is_main_term
                    continue
                self.pnb_used.append(pn)
                parent = psett['parent']
                if parent is not None or parent == '':
                    assert parent in param_names
                    assert parent != pn
                    if not parent in self.parameters_couplings:
                        self.parameters_couplings[parent] = []
                    self.parameters_couplings[parent].append(pn)
                    self.parameters_couplings[pn] = None
                else:
                    internn += main_dim
                    hsett = {}
                    hsett['model_settings'] = psett['model_settings']
                    hsett['model'] = psett['model']
                    
                    if is_output:
                        slices.append(sindx, sindx + main_dim)
                        sindx = main_dim
                    hsett['name'] = pn
                    model = self.get_head_model(
                        main_dim, hsett, False, not is_output
                    )
                    self.model_heads.append(model)

            dft_subact['dim'] = internn
            inter_model = self.get_head_model(
                main_dim, dft_subact, False, not is_output
            )
            self.terminal_model = CellHead(
                inter_model, self.model_heads, slices, is_output, 
                self.pnb_used, self.parameters_couplings
            ) #merger model
        else:
            self.terminal_model = lambda x: x

    def update_activation_hist(self, cell_fraction):
        '''cell_fraction: norm counts of cells activated for
            this terminal, the terminal has the counts, not the model'''
        self.activation_hist.shift_hist(
            {'hist': cell_fraction, 'hist_time': time.time()}, 0
        )


    def forward_term(self, values):
        '''**kwargs, because there could be more than one value'''
        '''returns dict of tensors'''
        return self.terminal_model(values)
