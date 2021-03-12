import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSample(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        return g.op('GridSample', x, grid, mode, padding_mode, align_corners)

    @staticmethod
    def forward(self, x, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        return F.grid_sample(x, grid, mode, padding_mode, align_corners)

