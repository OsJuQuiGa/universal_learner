# Forward function for terminals
# All of these forward fuctions need a terminal and the parameter names if given
# and only handle the input and output of data from or out of the system so beads
# or the chains handle the losses with respect this data
# terminal: object with the method to forward to
# param_names: parameters to get or save
import random
import torch
def _input_real(Hmemory, terminal, phase,**kwarg):
    media = terminal.get_media(param_names, **kwarg)
    return terminal.set_parameter_values(media)

def _io_no_real(Hmemory, terminal, phase, **kwarg):
    cat = terminal.base_counter_subcat
    return Hmemory.get_terminal_parameters(
        terminal.param_names, cat['io'],
        (cat['type'], cat['source']),
        terminal.device
    )
def _input_encoder(Hmemory, terminal, phase, **kwarg):
    # dream_noise_init: boolean for initialize the enconder with noise
    # to make a interpretation
    source = terminal.category_dict['source']
    if phase['wake']:
        typ = 'real' if terminal.has_real_expression else 'virtual'
        past = Hmemory.get_terminal_parameters(
            param_names, 'input', (typ, source), terminal.device
        )
    elif phase['dream_enc'] == 'noise':
        past = terminal.generate_random_parameter_values(param_names=param_names,**kwarg)
    #elif phase['enc_past'] == 'sleep_past':
    elif phase['dream_enc']  == 'old':
        old = Hmemory.get_terminal_parameters(
            param_names, 'input', ('critic', source), extra='inter_old'
        )
        past = torch.tensor(old[random.randint(0,len(old))], device=terminal.device)
    terminal.set_parameter_values(past, param_names = param_names)
    return past

def _input_dream(Hmemory, terminal, phase, **kwarg):
    # Dream forward used sparingly in wake state, it takes mostly
    # a interpretation of the encoder
    # wake: boolean for a wake chain, get the expression of the system instead
    # of the interpretation in the encoder
    if phase['dream_init'] == 'critic':
        source = terminal.category_dict['source']
        past = Hmemory.get_terminal_parameters(
            param_names, 'output', ('critic', source), terminal.device
        )
    elif phase['dream_init'] == 'virtual':
        past = Hmemory.get_terminal_parameters(
            param_names, 'output', ('virtual', source), terminal.device
        )
    else:
        past = _io_no_real(Hmemory, terminal, param_names, phase, **kwarg)
    return past
'''
def _output_gather(terminal, extra, list_outputs):
    #do a reduce operation over the list of outputs
    pass
'''
def _output_real(Hmemory, terminal, **kwarg):
    # set values of avg, sum or whatever (_output_common), then do this.
    parameters = terminal.express_parameters(param_names = param_names)
    return terminal.set_parameter_values(parameters)

def _output_encoder(Hmemory, terminal, **kwarg):
    # Saves the values expressed as an interpretation of the encoder
    # if the system has a real expression then it uses a sample from the enviroment
    # otherwise it uses am expression from the virtual terminal
    typ = 'real' if terminal.has_real_expression else 'virtual'
    device = terminal.parameters[terminal.param_names[0]].device
    #device = terminal.device
    sample = Hmemory.get_terminal_parameters(
            param_names, 'input', (typ, terminal.category_dict['source']), device
        )
    return terminal.parameters, sample

def _output_critic(Hmemory, terminal, **kwarg):
    # Does the same as the encoder, it returns a interpretation and param_loss pair for
    # Loss backprop, and another pair interpretation and a past interpretation, this last
    # pair is used as a residual loss for the gaussian variation of beads output and
    # finishing a critic cycle
    interpretation, sample = _output_encoder(
        Hmemory, terminal, param_names, extra, phase, **kwarg
    )
    past_interpretation = terminal.extra_parameters['interpretation']
    terminal.set_parameter_values(
        interpretation, extra = 'interpretation'
    )
    if random.random() < 0.001:
        terminal #TODO add to dream pool
    return interpretation, sample, past_interpretation


_forward_dict_fn={
    'input':{
        'real':_input_real,
        'dream':_input_dream, #Anti-real
        'virtual':_io_no_real,
        'encoder':_input_encoder,
        'critic':_input_encoder,
    },
    'output':{
        #'real': _output_real,
        #'dream': _io_no_real,
        #'virtual': _io_no_real,
        'real': None,
        'dream': None,
        'virtual': None,
        'encoder': _output_encoder,
        'critic': _output_critic,
    }
}