import torch


class ToPositive(torch.nn.Module):
    def __init__(self, mean, std, min_=0.):
        super().__init__()

        self.register_buffer("mean_over_std", torch.tensor(float(mean/std)))
        self.register_buffer("std", torch.tensor(float(std)))
        self.register_buffer("min_", torch.tensor(float(min_)))

    def forward(self, x):
        # for min=0, implements m+x*s for m+x*s > s, s*exp(m/s+x-1) else  (linear with mean and std until one std over zero)
        return self.std * (torch.nn.functional.elu(self.mean_over_std+x-1)+1) + self.min_
    
# maps values to a range (0, max) with mean max/2 and given approx std
class ToRange(torch.nn.Module):
    def __init__(self, max_, std):
        super().__init__()
        self.register_buffer("std_over_max", torch.tensor(float(std/max_)))
        self.register_buffer("max", torch.tensor(float(max_)))

    def forward(self, x):
        # 
        return self.max * torch.sigmoid(self.std_over_max*x)