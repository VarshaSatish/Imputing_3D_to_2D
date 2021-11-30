import torch
import pickle
import time
import copy
import sys
import os
from torch import nn
from torchsummary import summary
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from mpjpe import cal_mpjpe
from model import LiftModel
from data_gen import Data_Prep

cwd = os.getcwd()
print(cwd)
PATH = os.path.join(cwd, 'liftModel')
with open(os.path.join(cwd, 'data_test_lift.pkl'),'rb') as f:
    data = pickle.load(f)
   
###converting the dict type data to tensor
inputs_original = torch.tensor(data['joint_2d_1'])
targets_original = torch.tensor(data['joint_3d'])
focal_len = torch.tensor(data['focal_len_1'])

test_set = Data_Prep(inputs_original, targets_original)
test_generator = torch.utils.data.DataLoader(test_set, batch_size = 64, num_workers = 0,)
   
print('Test dataset length = ',len(test_set))

model = LiftModel(n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3)
model.load_state_dict(torch.load(PATH))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#loss function
criterion = cal_mpjpe()

running_loss = 0.0
for _, targs in enumerate(test_generator):
    inputs = targs[0]
    targets = targs[1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    running_loss += loss.item()
final_loss = running_loss/len(test_generator)
print(' Testing loss {:.4f}'.format(final_loss))