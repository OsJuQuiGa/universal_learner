import torch
import torch.nn as nn
from genericsocket import Socket
from terminals.genericterminal import Terminal

class Input(Socket):
    def __init__(self):
        super().__init__()
        self.io['io'] = 'Input'

class Real(Input, Terminal):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Real'

    def forward(self, *arg, parameters_names = '', **kwarg):
        self.get_media(parameters_names, kwarg)
        return self.values

class Virtual(Input):
    #This adds capacity to the model
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Virtual'

    def forward(self, *arg):
        self.load_values(counter={'io':'Output'})
        return self.values

class Encoder(Input):
    #takes the last media taken from real/dream and calculates the loss with respect a recent
    def __init__(self, wake=True):
        super().__init__()
        self.io['type'] = 'Encoder'

    def forward(self, wake=True):
        #parameters must be an array with type+source+name to search in pool
        if wake:
            past = self.load_values(counter={'io':'Output'})
        else:
            past = self.rand()
        return past

class Critic(Input):
    #takes the last media taken from real/dream and calculates the loss with respect a recent
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Critic'

    def forward(self, *arg):
        #parameters must be an array with type+source+name to search in pool
        past = self.load_values(counter={'io':'Output'})
        return past

class Dream(Input):
    #acts like virtual but its results are critized
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Dream'

    def forward(self, wake= True):
        #parameters must be an array with type+source+name to search in pool
        if wake:
            self.load_values(counter={'io':'Output'})
        else:
            self.load_values(counter={'io':'Output', 'type':'Encoder'}, preffix = 'interpretation')
        return self.values

input_classes = [
    Real,
    Virtual,
    Encoder,
    Critic,
    Dream
]
 
        
