import torch
from torch import nn

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x, y):
        a = (y-x)**2
        lossa = torch.sum(a, dim=1)
        return torch.sum(lossa) / x.size()[0]