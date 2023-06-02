from utils import check_key, iter_convert
import random
import os
import pickle
import torch
#Post-process settings


def post_process_settings(settings, acc_devices = {},
    wake_phase = 'all', only_expansion = False):
    '''
        Modifies the settings dict to add variables that can be
        automated based in the settings given, also variables that could
        otherwise be intensibly recalculated in various loops and are better
        store and calculated only once

        1) expand terminals makes clear which terminals are available for each
        type and number of terminals to use, it uses inclusive and exclusive masks
        also defines a mask later used in sampling the action that can be taken
        by the cell
        2) subaction process adds the dimension of the output, also defines the
        slice for the partial residual for each subaction head and each action per
        head, also the mask for the memory heads
        3) sets the diffusenes indeces

        settings: dict: base settings
        acc_devices: dict: acceleration devices to use, for later defining the masks
        wake_phase: str: sets the types to use in that phase
        only_expansion: bool: if only post-process the expansion part 1)
        ---return: settings: dict: post-processed settings
    '''
    #-------------------EXPAND TERMINALS----------------------        
    
    mask_inclusive = settings['terminal_sockets_mask']['inclusive']
    mask_exclusive = settings['terminal_sockets_mask']['exclusive']

    all_types = settings['terminal_types'] + settings['special_terminal_types']
    type_avail = {}
    if wake_phase in settings['phase_terminal_types']:
        phase_types = settings['phase_terminal_types'][wake_phase]
    else:
        phase_types = all_types

    settings = iter_convert(settings)

    for io in settings['terminal_io']:
        settings['terminal_expansion'][io] = []
        settings['terminal_expansion_full'][io] = [] #with dream and wake types
        settings['terminal_expansion_fast'][io] = []
        settings['all_sources'][io] = []
        for order, cls_name in enumerate(settings['terminal_order']):
            cls_settings = settings['terminal_sockets'][cls_name]
            #cls_settings['expansion'] = {} #if 'expansion' in cls_settings else cls_settings['expansion']
            
            if check_key(cls_settings, 'no_use'):
                continue
            base_name = cls_settings['name'] if 'name' in cls_settings else ''
            n_obj = cls_settings['number'] if 'number' in cls_settings else 1
            #cls_settings['expansion'][io] = []
            for i in range(n_obj):
                name = base_name
                name += '' if n_obj == 1 else str(i)
                term_name = cls_name + name
                types_used = []
                types = []
                #cls_settings['expansion'][io][i] = {}
                settings['all_sources'][io].append(term_name)
                type_avail[term_name] = []
                for tp in all_types:
                    if cls_name in mask_exclusive:
                        if io in mask_exclusive[cls_name]:
                            if tp in mask_exclusive[cls_name][io] or mask_exclusive[cls_name][io][0] == 'all':
                                continue
                    types_used.append(tp)

                if cls_name in mask_inclusive:
                    if io in mask_inclusive[cls_name]:
                        for intp in mask_inclusive[cls_name][io]:
                            if not intp in types_used:
                                types_used.append(intp)

                for tp in types_used:
                    tp_idx = all_types.index(tp)
                    order_i = tp_idx*1e3 + order*1e6 + i
                    to_append = (cls_name, tp, term_name, i, order_i)
                    if tp in phase_types:
                        settings['terminal_expansion'][io].append(to_append)
                        if io == 'Output':
                            type_avail[term_name].append(tp)
                    settings['terminal_expansion_full'][io].append(to_append)
        settings['terminal_expansion'][io].sort(key=lambda x: x[-1])
        settings['terminal_expansion_full'][io].sort(key=lambda x: x[-1])
        settings['terminal_expansion'][io] = [tuple(x[:-1]) for x in settings['terminal_expansion'][io]]
        settings['terminal_expansion_full'][io] = [tuple(x[:-1]) for x in settings['terminal_expansion'][io]]
        #for i, ttup in enumerate(settings['terminal_expasion'][io]):
        #    settings['terminal_sockets'][ttup[0]]['expansion'][io][ttup[3]][ttup[1]] = i

        #-----------------------------DIFFUSSENESS--------------------------------------
        settings['diffuseness_dims'][io] = len(settings['terminal_expansion_full'][io])
    settings['diffuseness_dim'] = settings['diffuseness_dims']['Output']
    settings['n_all_cells'] = settings['n_cells']+2*settings['diffuseness_dim']

    full_exp = settings['terminal_expansion_full']
    assert full_exp['Input'] == full_exp['Output']
    #make sure these are equal for now, much easier to implement diffuseness routing
    routing_strength_matrix = []
    type_mult = settings['diff_type_routing_mult']
    source_mult = settings['diff_source_routing_mult']
    for in_tup in full_exp['Input']:
        routing_strength_matrix.append([])
        for out_tup in full_exp['Output']:
            in_type = in_tup[1]
            in_source = in_tup[0]
            out_type = out_tup[1]
            out_source = out_tup[0]
            factor = 1
            if in_type in type_mult:
                if out_type in type_mult[in_type]:
                    factor *= type_mult[in_type][out_type]
            #if in_source == out_source: #this is for diffuseness simimilarity
            #    factor *= source_mult[in_type]
            routing_strength_matrix[-1].append(factor)
        routing_strength_matrix[-1] = torch.tensor(routing_strength_matrix[-1])
    routing_strength_matrix = torch.stack(routing_strength_matrix)
    settings['routing_strength_matrix'] = routing_strength_matrix

    settings['all_types'] = all_types
    settings['types_available'] = type_avail
    real_types = settings['real_types']
    settings['all_types_virtual'] = [typ for typ in list(all_types) if not typ in real_types] 

    if only_expansion:
        return settings
    #type masks for phases are now dealt by the cell at __init__ level
    #using tensor operations that are faster.

#------------------------------DEVICE MEMORY-----------------------------------
    for device_set in settings['memory']['devices'].values():
        if device_set['type'] == 'CPU':
            device_set['max_mem'] = int(open('/proc/meminfo').read()[9:25])
        elif device_set['type'] == 'GPU':
            #mem_f = open(
            #    '/sys/bus/pci/devices/%s/mem_info_vram_total' % device_set['id']
            #, 'r')  FOR AMD CARDS
            # mem = int(mem_f.readline())
            mem = 1e10
            device_set['max_mem'] = mem
#-----------------------------ROUTING MASKS------------------------------------
    n_connections = settings['actions']['where']['dim']
    conn_masks = {
        'cell_type': torch.zeros(n_connections, 
            len(settings['cell_types']) , device='cpu'),
            #cell or terminal or kill
        'type': torch.zeros(n_connections, 
            len(all_types), device='cpu'), #for terminals
        'source': torch.zeros(n_connections, 
            len(settings['all_sources']['Output']), device='cpu') #for terminals
    } #used for specifing which terminals(connections) are possible in phase
    settings['conn_masks'] = conn_masks
    
    return settings