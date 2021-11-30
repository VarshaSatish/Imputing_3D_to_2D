# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:34:01 2021

@author: Varsha S
"""

import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_layer, dropout):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(dropout, inplace = True)
            )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(dropout, inplace = True)
            )

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.linear2(x)
        x = x + residual
        return x

class LiftModel(nn.Module):
    def __init__(self, n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3):
        super(LiftModel, self).__init__()
        self.blocks = []
        self.n_blocks = n_blocks
        self.in_layer = nn.Linear(15*2, hidden_layer)
        for i in range(self.n_blocks):
            self.blocks.append(ResidualBlock(hidden_layer, dropout))
        self.blocks = nn.ModuleList(self.blocks)
        self.out_layer = nn.Linear(hidden_layer, output_nodes)

    def forward(self, x):
        x = self.in_layer(x)
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.out_layer(x)
        return(x)

#def residual():
#    model = LiftModel(n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3)
#    return model