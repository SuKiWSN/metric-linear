from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, model, use_matrix):
        super(Model, self).__init__()
        # self.matrix = nn.Parameter(torch.eye(100))
        self.matrix = nn.Parameter(torch.randn((100, 100)))
        self.norm = nn.LayerNorm(normalized_shape=[100])
        self.model = model
        self.use_matrix = use_matrix

    def forward(self, x):
        x = self.model(x)
        if self.use_matrix:
            matrix = self.norm(self.matrix)
            x = x @ matrix
        return x
