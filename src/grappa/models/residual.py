import torch

class ResidualDenseLayer(torch.nn.Module):
    def __init__(self, activation, module_fct, *module_args):
        super().__init__()
        self.module_skip = module_fct(*module_args)
        self.module1 = module_fct(*module_args)
        self.module2 = module_fct(*module_args)
        self.activation = activation


    def forward(self, h):
        h_skip = self.module_skip(h)
        h = self.module1(h)
        h = self.activation(h)
        h = self.module2(h)
        return self.activation(h_skip+h)
