autoencoder ={
    'terminal_sockets':{
        'Retina':{
            'views':[1,4,16],
            'channels':32
        },
        'SystemAudio':{
            'channels':32
        }
    },
    'terminal_order':[
        'Retina',
        'SystemAudio',
    ],
    'subactions':{
        'what':{
            'modes':{
                'default':[
                    {'name': 'residue_decay', 
                    'out_dim':1, 'in_dim': 64, 
                    'activation':'sigmoid', 'loss': 'regression_n'},
                ]
            },
            'mode_preset':'default',
            'internn': 96,
            'use': False,
        },
        'where':{
            'modes':{
                'binary':[
                    {'name': 'do', 'out_dim':3, 'in_dim': 64, 'loss': 'categorical'},
                    {'name': 'bin_routing', 'out_dim':8, 'in_dim': 160, 'loss': 'categorical_binary'},
                    {'name': 'types', 'out_dim':None, 'in_dim': 64, 'loss': 'categorical'}, #out_dim defined by ntypes
                    {'name': 'sources', 'out_dim':None, 'in_dim': 64, 'loss': 'categorical'}, #out_dim defined by nsources
                ]
            },
            'internn':256
        }
    },
    'terminal_subaction':{
        'internn':256,
    },
    'main_model_parameters':{
        'layers':5,
        'in_dim':512,
    }, #assumes MLP, for now
    
    #'cell_parameters':{
    #    'MLP':{
    #        'layers':4,
    #        'in_dim':512
    #    }
    #},
    'cell_limits':{
        'branching':{'max':4},
        'deepness':{'max':30}
    },
    'debug_mode': False,
    'memory':{
        'experiment':'autoencoder',
        'experiments_folder': '/mnt/D/Recovery/Experiments/',
        'filter_time_freq':32,
        'update_checkpoints': 7500,
        'devices':{
            'cpu':{'soft_limit':1024},
            'cuda:0':{'soft_limit':256},
            'cuda:1':{'soft_limit':256},
        }

    }
}
test = {
    'terminal_sockets':{
        'Test':{}
    },
    'terminal_order':[
        'Test'
        'Retina',
        'SystemAudio',
    ],
}
onlytest = {
    'terminal_order':[
        'Test'
    ]
}
none = {}
agent = {
    'Retina':{
        'views':[1,4,16],
        'channels': 32,
        'size': 64,
    },
    'SystemAudio':{
        'channels':32,
        'fs':16000
    },
    'terminal_order':[
        'Retina',
        #'SystemAudio',
        'SystemKeyboard',
        'SystemPointerClick',
        'SystemPointerLocation'
    ],
}
