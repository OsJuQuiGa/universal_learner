import torch
import torch.nn as nn
from genericsocket import Socket

class Output(Socket):
    def __init__(self):
        super().__init__()
        self.io['io'] = 'Output'

class Real(Output):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Real'

    def forward(self, *arg):
        self.express_values()
        return self.get_parameters()

class Virtual(Output):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Virtual'

    def forward(self, *arg):
        self.save_values()
        return self.get_parameters()

class Encoder(Output):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Encoder'

    def forward(self, *arg):
        self.save_values(preffix = 'interpretation')
        param_int = self.get_parameters()
        if self.has_expression:
            self.get_media()
            self.get_media_values()
            self.save_values()
        else:
            self.load_values(counter = {'type':'Virtual'})
        param_loss = self.get_parameters()
        return param_int, param_loss
   
class Critic(Output):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Critic'

    def forward(self, *arg):
        param_critic, param_loss = Encoder.forward(self, interpretation)
        self.load_values(counter = {'type':'Encoder'}, preffix = 'interpretation')
        param_int = self.get_parameters()
        return param_critic, param_int, param_loss #interpretation is the actual one, the self one is the past one

class Dream(Output):
    def __init__(self):
        super().__init__()
        self.io['type'] = 'Dream'

    def forward(self, *arg):
        self.save_values()
        return self.values

output_classes = [
    Real,
    Virtual,
    Encoder,
    Critic,
    Dream
]