settings = {
    'terminal_sockets':{ #expand for output, 'expasion':types
        'Retina':{
            'imitate_master': True,
            'batch_size': 1, #1 to 3
            'views':[1,4],
            'featchannels': 8,
            'channels': 3,
            'size': 32,
            'target_flat_dim': [2048,1024,512],
            'use_affine_transform': True,
            'rate':5,
            'receptors':8,
        },
        'SystemSensors':{
            'imitate_master': False,
            'batch_size': 1,
            'sensors':['cycletime', 'daytime', 'routings', 'ticking', 'losses']
        }, #like cycletime, battery, daytime, temperatures, memory usage
        'SystemKeyboard':{
            'imitate_master': True,
            'batch_size': 1, 
            'rate':20},
        'SystemPointer':{
            'imitate_master': True,
            'batch_size': 1,
            'rate':20},
        'MemoryBank':{
            'imitate_master': False,
            'batch_size': 1, 'number':4},
        #'MemoryBankWithMemoryModule':{'number':4, 'model_settings':{'dims':256}}, long term memory bank
        'SystemAudio':{
            'imitate_master': True,
            'batch_size': 1,
            'channels': 2,
            'receptors':8,
            'div_note': 3,
        },
        'Test':{
            'imitate_master': True,
            'batch_size': 1,
        }
    },
    'terminal_order':[ #The ones that are going to be used and ordered
        #The ones with real expression have to go first
        'Retina',
        'SystemAudio',
        #SystemBasicIO
        #'SystemPointerLocation',
        #'SystemKeyboard',
        #'SystemPointerClick',
    ],
    'terminal_sockets_mask':{
        'exclusive':{ #remove only these
            'MemoryBank':{'Input':['Real', 'Dream'], 'Output':['Real', 'Dream']},
            'InternalTerminal':{'Input':['Real', 'Dream'], 'Output':['Real', 'Dream']},
        },
        'inclusive':{
        }
    },
    'terminal_types':[
        #'real', 'dream', 'virtual', 'virtualdream', 'encoder', 'critic'   #used for all
        'Real', 'Critic', #'Dream', 'DreamCritic'
        #'real_dream', 'virtual', 'critic'
    ],
    'types_reward_track':[
        'Real', 'Critic', 'Dream', 'DreamCritic'
    ],
    'shared_terminal_types':{
        'Input':{
            'Real':{'Real', 'Critic'},
            'Dream':{'Dream', 'DreamCritic'},
            'Critic':{},
            'DreamCritic':{},
            'Virtual':{'Virtual'},
        },
        'Output':{
            'Real':{'Real', 'Virtual'},
            'Virtual':{'Real', 'Virtual'},
            'Critic':{'Critic'}, #same source
            'DreamCritic':{'DreamCritic'}, #same source
            'Encoder':{'Encoder'} #same source
        }
    },
    'real_types':{'Real', 'Encoder', 'Critic'},
    'phase_terminal_types':{
        'wake':['Real', 'Critic', 'Virtual'], 
        'dream':['Dream', 'DreamCritic', 'Critic'],
    },
    'phase_terminal_types_to_learn':{
        'wake':['Real', 'Critic', 'Virtual'],
        'dream':['Dream', 'DreamCritic'],
    },
    'special_terminal_types':[
    ],
    'terminal_expansion':{ #for input(order of execution) and output(attention mechanism)
        #io:(source, number, type, order_i)
    },
    'terminal_expansion_fast':{
        #(io:(type, term_name))
    },
    'terminal_expansion_full':{
        #(io:(type, term_name))
    },
    'terminal_io':[
        'Input','Output'
        #'Output','Input'
    ],
    'other_diffdim':[
        'Kill',
        #'Resume', if it starts without input forward_fn
    ],
    'all_sources':{ #it has the term_name
        #all sources with the number of objects to use, io:source, post
    },
    'all_types':[

    ],
    'all_types_virtual':[
        
    ],
    'cell_types':[
        'Cell',
        'Terminal',
        'Kill',
    ],
    'batch_imitation': True,
    'day_big_loss_max':10.0,
    'utc_hour': -5,
    'exp_t0': 5.0,
    'n_threshold': 2e5,
    'noise_dream_iter': 2000,
    'phase_change_t': 500, #iterations to reach "equilibrium", after this qty lr grows
    'back_all': True,
    'reward_trainded_threshold': 1e5,
    'mutate_param':{
        'bottom': 0.05,
        'take_best': 0.25,
        'reset': 0.005,
        'point': 0.50, #classic mutation of weights and bias
        'duplicate': 0.25, # Duplicate the fittest cell
        'crossover_p':{
            'magnitude': 0.30, 
            #takes the weight magnitude and bias of each layer
            #then set them to the layers of the other model in 
            #the same order of those layers' magnitude
            'magnitude_groups':0.30,
            'swap_groups': 0.10,
            #classic crossover of weights, with random points
            #setting pi of cutting to 0.001 and slowly increasing
            #the pi of cutting
            'swap_layers':0.10,
            #swaps the layers of each model
        },
    },
    'diff_type_routing_mult':{
        'Real':{'Real':1.5, 'Virtual': 1.5},
        'Virtual':{'Real':1.25, 'Virtual': 1.25, 'Dream':1.25},
        'Critic':{'Critic':1.5},
        'Dream':{'Dream':1.5, 'Virtual':1.5},
        'DreamCritic':{'Critic':0.1, 'DreamCritic':1.5}
    },
    'diff_source_routing_mult':{
        'Real':1.25,
        'Virtual':1.25,
        'Critic': 2.0,
        'Dream': 1.25,
        'DreamCritic':2.0,
    }, #NOPEEEE
    'terminal_loss_reduction_function':'softmax',
    'loss_lambdas':{
        'types': 1.0,
        'reward_critic':0.5,
        'reward_value':0.5,
        'curiosity':0.25,
        'diffuseness':0.01,
        'encoder':0.5,
        'critic':0.5,
        'memory':0.01,
        'entropy':0.005,
        'time': 5.0,
        'boundary': 0.0,
        'not_ended':10.0,
    },
    'screens':{
        'slave': [{'width':1280, 'height':720}],
        'master': [{'width': 2560, 'height':1080}] #use topleft screen crop
    },
    'screen_dims':['width', 'height'],
    'batch_size':1,
    'filter_model_param': {
        'in_dim': 96,
        'layers': 2,
    },
    'actions':{
        'where':{
            'model': 'MLP',
            'routings': 2,
            'connections': 32, 
            'dim': 64,
            'model_settings':{
                'MLP':{
                    'layers':4
                },
                'HistTransformer':{
                    'nheads':2,
                    'num_encoder_layers':1,
                    'num_decoder_layers':1,
                    'dim_feedforward': 256,
                    'mask_decoder_attn': True,
                    'use_time_encoding': 'all'
                }
            }
        },
        'memory':{
            'write_ratio':0.1,
            'model': 'MemTransformer',
            #'dim': None,
            'mem_len': 16,
            'model_settings':{
                'MemTransformer':{ #Static memory
                    'nheads':2,
                    'num_encoder_layers':2,
                    'num_decoder_layers':2,
                    'dim_feedforward': 256,
                    'use_time_encoding': 'enc',
                    'mask_decoder_attn': True,
                    'mem_len':'_.actions.memory.mem_len',
                    'seq_len':'_.seq_len'
                },
                'MemMLP':{ #NOT done
                    'layers':4,
                    'mem_len':'_.actions.memory.mem_len'
                },
            },
        },
    },
    'main_model_parameters':{
        'CellMLP':{
            'layers':4,
            'in_dim':'_.main_model_parameters.general.dim',
        },
        'HistTransformer':{
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'nheads': 3,
            'dim': '_.main_model_parameters.general.dim',
            'dim_feedforward': 512,
            'seq_len': '_.seq_len',
            'mask_decoder_attn': True,
            'use_time_encoding': 'all',
            'relative_modulator_settings':{
                'use': True,
                'layers': 2,
                'singular': True,
            },
        },
        'general':{
            'dim': 64
        }
    },
    'msg_dim': '_.main_model_parameters.general.dim',
    'seq_len': 32,
    'main_model':'CellMLP',
    'subactions_order':{
        'pretraverse':['memory'],
        'innertraverse':[
            'where'
        ]
    },
    'subactions_slices':{
        #index range of each subsub action of the subactions,
        #for what these are Lr, residue...etc, when there are actions and not
        #heads for each action
    },
    'no_traversed_penalty_div':{
        'terminal': 2,
        'cell': 5,
    },
    'cell_limits':{
        'routing':{'max':5000},
        #'branching':{'max':4}, 
        #'deepness':{'max':15},
        'mutation':{'max': 0.025, 'cell_ratio':0.75},
    },
    'cell_model_parameters':{
        'use_subactions':['where'],
        'use_metatraining':False,
        'use_filtering': False,
    },

    #'cell_arguments':{
    #    'lr':0.0001,
    #    'optimizer': 'SGDdom',
    #    'decay':0.01,
    #    'loss_fn': 'MSE',
    #}
    'terminal_subaction':{
        'internn':192,
        'n_replicas':1,
        'model': 'MLP',
        'model_settings':{
            'MLP':{}
        },
    },
    'debug_mode': True,
    'cycle_execution': 'sequential',
    'internal_values_mode': 'residual',
    'antilearn':{ #set the loss to negative values
        'threshold_min': 0.25, #adds
        'threshold_max': 10.0,#removes to activation
        'activation': 100,
        'duration': 50,
    },
    'n_cells': 128,
    'diffuseness_dims':{},
    'diffuseness_masks':{},
    'internal_values_components': { 
        'time_encoding':['minute', 'hour', 'day'],
        'tick_encoding':[16, 256],
        'branching': [10], #10th part
        'deepness': [10,100], #100th and 10th part
        'traverse_encoding':{}
    },
    'internal_values_components_order':[
        'last_cell_maturity', #1
        #'n_traverse', #1
        'terminal_order', #1
        #'time_encoding', #4 mix time and tick encoding in transformer
        #'tick_encoding', #4
        'avg_time', #1
        #'branching', #1
        #'deepness', #2
        #'past_loss', #1 stress
        #'traverse_encoding', #12
        #'diffuseness', #15
        #'internal_terminal', #32
    ],
    'internal_values_dim':{}, #defined in post processing
    'memory':{
        'experiment': 'test',
        'experiments_folder': './experiments/',
        'checkpoint_path': '_cell_checkpoint.pth',
        'filter_time_freq':16,
        'update_checkpoints': 64,
        'used_cnt':5,
        'nouse_cnt':5,
        'devices_distribution':'iocells', #['iocells', 'split'(broken)]
        'devices':{
            'cpu':{
                'soft_limit':32,
                'type':'CPU', 
                'max_mem': 32000000,
                'use': True,
                'ttype':'cpu',
                'idx':-1
            },
            'cuda:0':{
                'soft_limit':16, 
                'id': '0000:0c:00.0', 
                'type': 'GPU', 
                'max_mem': 8573157376,
                'use': True,
                'ttype':'cuda',
                'idx':0
            },
        }
    }
}
