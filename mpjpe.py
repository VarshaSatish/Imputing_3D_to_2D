# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:35:01 2021

@author: Varsha
"""

# function that calculates mjpje between a batch of poses. 
# For our experiments one of them should be the GT poses 
# and the other be their corresponding predicted ones.
# Note: the function expects Tensors.
# of the form (batch_size, 15*3) or (batch_size, 15, 3)
import torch
from torch import nn
import numpy as np
class cal_mpjpe(nn.Module):
    def __init__(self):
        super(cal_mpjpe, self).__init__()
        
    def forward(self, pose_1, pose_2, avg=True):
        n_joints = 15
        batch_size = pose_1.shape[0]
#        pose_1 = np.copy(pose_1.reshape(batch_size, n_joints, 3))
#        pose_2 = np.copy(pose_2.reshape(batch_size, n_joints, 3))
#        pose_1 = torch.reshape(pose_1, [batch_size, n_joints, 3])
#        pose_2 = torch.reshape(pose_2, [batch_size, n_joints, 3])
        
        diff = pose_1-pose_2
        diff_sq = diff ** 2
        dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
        dist_per_sample = torch.mean(dist_per_joint, axis=1)
        if avg is True:
            dist_avg = torch.mean(dist_per_sample)
        else:
            dist_avg = dist_per_sample
        return dist_avg