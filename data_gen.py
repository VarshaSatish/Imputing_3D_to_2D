# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:16:41 2021

@author: Varsha
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data.dataset import Dataset

class Data_Prep(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay
        self.size = len(datax)

    def __getitem__(self, index):
        # stuff
        inputs = self.datax[index].clone().detach().requires_grad_(True)
        targets = self.datay[index].clone().detach().requires_grad_(True)
        inputs = torch.transpose(torch.reshape(inputs,(-1, 1)), 0, 1)
        targets = torch.transpose(torch.reshape(targets,(-1, 1)), 0, 1)
        return (inputs, targets)

    def __len__(self):
        return self.size
