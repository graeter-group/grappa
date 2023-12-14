#%%
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


        if not isinstance(mean/std, torch.Tensor):
            mean_over_std = torch.tensor(float(mean/std)).float()
        else:
            mean_over_std = (mean/std).float()
        
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(float(std)).float()
        else:
            std = std.float()

        if learnable_statistics:
            self.register_parameter("mean_over_std", torch.nn.Parameter(mean_over_std))
            self.register_parameter("std", torch.nn.Parameter(std))

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

        if isinstance(std, torch.Tensor):
            std = std.float().detach()

        if isinstance(max_, torch.Tensor):
            max_ = max_.float().detach()

        if not isinstance(std/max_, torch.Tensor):
            std_over_max_ = torch.tensor(float(std/max_)).float()
        else:
            std_over_max_ = (std/max_).float()

        if not isinstance(max_, torch.Tensor):
            max_ = torch.tensor(float(max_)).float()
        else:
            max_ = max_.float()

        if learnable_statistics:
            self.register_parameter("std_over_max", torch.nn.Parameter(std_over_max_.clone()))
        else:
            self.register_buffer("std_over_max", std_over_max_)
            
        self.register_buffer("max", max_)


    def forward(self, x):
        # Avoid inplace operations by cloning the input tensor. For some reason this is needed if gradients are computed several times (also with retain_graph=True)
        x = x.clone()
        scaled_x = self.std_over_max * x
        sigmoid_x = torch.sigmoid(scaled_x)
        output = self.max * sigmoid_x
        return output
    

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    std = torch.tensor([1.]).item()
    model = ToRange(1, std, True)

    x = torch.randn(100)
    y = torch.abs(torch.randn(100))/4.

    plt.scatter(x,y)
    plt.scatter(x, model(x).detach())
    # %%
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for i in range(100):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print(loss.item())
    # %%
    plt.scatter(x,y)
    plt.scatter(x, model(x).detach())
    # %%
