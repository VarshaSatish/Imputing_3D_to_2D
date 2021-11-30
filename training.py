# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:59:49 2021

@author: Varsha S
"""

import torch
import pickle
import time
import copy
import sys
from torch import nn
from torchsummary import summary
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import argparse

from mpjpe import cal_mpjpe
from model import LiftModel
from data_gen import Data_Prep

parser = argparse.ArgumentParser()
parser.add_argument('-n', type = int , help='num_epochs', default = 200)
parser.add_argument('-l', type = float, help='lr', default = 0.001)
parser.add_argument('-b', type = int, help='batch_size', default = 64)
args = parser.parse_args()

#loading the pkl file to get data
with open(r'/home/misc/RnD_project/CV_new/data_train_lift.pkl','rb') as f:
    data = pickle.load(f)

##converting the dict type data to tensor
inputs_original = torch.tensor(data['joint_2d_1'])
targets_original = torch.tensor(data['joint_3d'])
focal_len = torch.tensor(data['focal_len_1'])

txtfile = '/home/misc/RnD_project/CV_new/train_logs.txt'
feature_extract = True
filepath = '/home/misc/RnD_project/CV_new/liftModel'
train_split = 0.7
model_lm = LiftModel(n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3)

# Detect if we have a GPU available and Send the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_lm.to(device)
# Print the model instantiated
print(model_lm)

##Summary of the model
#y = (1, 30)
#summary(model_lm, y)

def run_epoch(data, model, optimizer, scheduler, epoch_no, batch_size = 64, split = 'train'):

    #data pre-processing
    data_len = len(data['joint_2d_1'])
    train_size = int(train_split * data_len)
    valid_size = data_len - train_size
    train_datasetx, valid_datasetx = torch.split(inputs_original, [train_size, valid_size])
    train_datasety, valid_datasety = torch.split(targets_original, [train_size, valid_size])

    #dataloaders
    train_set = Data_Prep(train_datasetx, train_datasety)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size = batch_size, num_workers = 0,)
    valid_set = Data_Prep(valid_datasetx, valid_datasety)
    valid_generator = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, num_workers = 0,)

    print('Training dataset length = ',len(train_datasetx))
    print('Validation_dataset length = ',len(valid_datasetx))

    since = time.time()
    train_loss_history = []
    val_loss_history = []
    best_loss = 10000

    for epoch in range(epoch_no):
        print('Epoch {}/{}'.format(epoch, epoch_no - 1))
        print('-' * 10)
        with open(txtfile, "a") as text_file:
            text_file.write("Epoch number: %s :-" %epoch)

        running_loss = 0.0
        # training phase
        i = 1
        print("     \n")
        for _, targs in enumerate(train_generator):
            inputs = targs[0]
            targets = targs[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.train()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
              # backward + optimize only if in training phase
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_generator)
        train_loss_history.append(epoch_loss)
        print(' Training loss {:.4f}'.format(epoch_loss))
        with open(txtfile, "a") as text_file:
            text_file.write("Training loss: %s " %epoch_loss)

        #validation phase
        running_loss = 0.0
        model.eval()

        for _, targs in enumerate(valid_generator):
            inputs = targs[0]
            targets = targs[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            running_loss += loss.item()

        epoch_loss = running_loss / len(valid_generator)
        val_loss_history.append(epoch_loss)
        print('Validation loss {:.4f}'.format(epoch_loss))
        with open(txtfile, "a") as text_file:
            text_file.write(" Validation loss: %s \n" %epoch_loss)

        scheduler.step()
        # deep copy the model
        if  epoch_loss < best_loss:
            best_loss = epoch_loss
            epoch_val = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('\n Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('\n Best val loss: {:4f} at epoch: {}'.format(best_loss, epoch_val))

    with open(txtfile, "a") as text_file:
        text_file.write(" \n Best val loss: {}, at epoch: {}".format(best_loss, epoch_val))
        text_file.write(' \n Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), filepath)
    #model.load_state_dict(best_model_wts)
    return model, val_loss_history, train_loss_history

params_to_update = model_lm.parameters()

#loss function
criterion = cal_mpjpe()
#optimizer and scheduler
steps = 10
decayRate = 0.5
lr = args.l
optimizer_ft = optim.Adam(params_to_update, lr)
#scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, steps)
scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma = decayRate)
model_lm, val_loss, train_loss = run_epoch(data, model_lm, optimizer_ft, scheduler_ft, epoch_no = args.n, batch_size = args.b, split = 'train')

#plotting the loss curves
num_epochs = len(val_loss)
x = range(0,num_epochs)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,train_loss, label='train_loss')
ax.plot(x,val_loss, label='val_loss')
leg1 = ax.legend(loc='lower left')
ax.set_ylabel('Loss')
ax.set_xlabel('epoch')
plt.savefig('{}_loss_plot.png'.format(lr), format='png')