"""
Final layers for models where we wish to restrict the output to a certain range. Current implementation is for the ranges (0, inf) [ToPositive] and (0, max) [ToRange].
For the infinite case, we use the exponential linear unit (ELU) to map the output to (0, inf), without having vanishing (in the negative direction) or exploding (in the positive direction) gradients (as one would have when simply using the exponential function).
For the finite case, we use the sigmoid function to map the output to (0, 1) and then multiply by max to map to (0, max).
In both cases, mean and standard deviation of the output are used to scale and shift the functions such that an input with normal distribution would yield the same statistics. This enables the previous layers to output values in the range of a normal distribution, which was shown to be advantageous (as in batch normalization, https://arxiv.org/abs/1502.03167).
"""
import torch


class ToPositive(torch.nn.Module):
    """
    Maps values to (0, inf) with statistics of given (mean, std) if the input is normal distributed using the exponential linear unit (ELU).

    Final layers for models where we wish to restrict the output to a certain range. Current implementation is for the ranges (0, inf) [ToPositive] and (0, max) [ToRange].
    For the infinite case, we use the exponential linear unit (ELU) to map the output to (0, inf), without having vanishing (in the negative direction) or exploding (in the positive direction) gradients (as one would have when simply using the exponential function).
    For the finite case, we use the sigmoid function to map the output to (0, 1) and then multiply by max to map to (0, max).
    In both cases, mean and standard deviation of the output are used to scale and shift the functions such that an input with normal distribution would yield the same statistics. This enables the previous layers to output values in the range of a normal distribution, which was shown to be advantageous (as in batch normalization, https://arxiv.org/abs/1502.03167).
    If learnable_statistics is set to True, mean and std deviation are learnable parameters, initialized with the given values.
    """
    def __init__(self, mean, std, min_=0., learnable_statistics=False):
        """
        The argument mean is actually mean - min, i.e. the distance from the minimum to the mean!
        """
        super().__init__()

        if learnable_statistics:
            self.mean_over_std = torch.nn.Parameter(torch.tensor(float(mean/std)))
            self.std = torch.nn.Parameter(torch.tensor(float(std)))

        else:
            self.register_buffer("mean_over_std", torch.tensor(float(mean/std)))
            self.register_buffer("std", torch.tensor(float(std)))

        self.register_buffer("min_", torch.tensor(float(min_)))

    def forward(self, x):
        """
        for min=0, implements m+x*s for m+x*s > s, s*exp(m/s+x-1) else  (linear with mean and std until one std over zero)
        """
        return self.std * (torch.nn.functional.elu(self.mean_over_std+x-1)+1) + self.min_
    
class ToRange(torch.nn.Module):
    """
    Maps values to a range (0, max) with mean==max/2 and given approx std if the input is normal distributed.

    Final layers for models where we wish to restrict the output to a certain range. Current implementation is for the ranges (0, inf) [ToPositive] and (0, max) [ToRange].
    For the infinite case, we use the exponential linear unit (ELU) to map the output to (0, inf), without having vanishing (in the negative direction) or exploding (in the positive direction) gradients (as one would have when simply using the exponential function).
    For the finite case, we use the sigmoid function to map the output to (0, 1) and then multiply by max to map to (0, max).
    In both cases, mean and standard deviation of the output are used to scale and shift the functions such that an input with normal distribution would yield the same statistics. This enables the previous layers to output values in the range of a normal distribution, which was shown to be advantageous (as in batch normalization, https://arxiv.org/abs/1502.03167).
    If learnable_statistics is set to True, the std deviation is a learnable parameter, initialized with the given value.
    """
    def __init__(self, max_, std, learnable_statistics=False):
        super().__init__()

        if learnable_statistics:
            self.std_over_max = torch.nn.Parameter(torch.tensor(float(std/max_)))
        else:
            self.register_buffer("std_over_max", torch.tensor(float(std/max_)))
            
        self.register_buffer("max", torch.tensor(float(max_)))

    def forward(self, x):
        # NOTE: thee is an error, it should be
        # return self.max * torch.sigmoid(self.std_over_max*x)
        return self.max * torch.sigmoid(self.std_over_max*x)