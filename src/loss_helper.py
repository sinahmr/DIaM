##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .rmi_loss import RMILoss

from .lovasz_loss import lovasz_softmax_flat, flatten_probas



# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        # if 'loss' in self.configer and 'params' in self.configer['loss'] and 'ce_weight' in self.configer['loss']['params']:
        #     weight = self.configer["loss"]["params"]["ce_weight"]
        #     weight = torch.FloatTensor(weight).cuda()
        # if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
        #     weight = self.configer.get('loss', 'params')['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()

        reduction = 'mean'
        if 'loss' in self.configer and 'params' in self.configer['loss'] and 'ce_reduction' in self.configer['loss']['params']:
            reduction = self.configer['loss']['params']['ce_reduction']
        # if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
        #     reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        if 'loss' in self.configer and 'params' in self.configer['loss'] and 'ce_ignore_index' in self.configer['loss']['params']:
            ignore_index = self.configer['loss']['params']['ce_ignore_index']
        
        # print(f'ignore index......{ignore_index}')
        # print(f'weight index......{weight.shape}')
        # print(f'reduction index......{reduction}')
        # self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            # print(f'target shape dimension ....{targets[0].shape}')
            # targets = targets[0].squeeze(1)
            targets = targets[0].clone().unsqueeze(1).float()
            # print(f'type of input object.....{type(inputs)} and type of target object....{type(targets)}')
            # for key, value in inputs.items():
            #     print(f"{key}: {value.shape}")
            
            # target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            # print(f"the segmentation......{inputs['seg'].shape} and {inputs['seg'].size(2)}" )
            # print(f'INPUT IN FORWARD OF LOSS {inputs.shape}  {inputs.size(2)}')
            # target = self._scale_target(targets[0], (inputs['seg'].size(2), inputs['seg'].size(3)))
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            # INPUT TO LOSS SHAPE......torch.Size([20, 81, 53, 53])
            # print(f"INPUT TO LOSS SHAPE......{inputs['seg'].shape}")
            # TARGET TO LOSS SHAPE......torch.Size([1, 53, 53])
            # print(f"TARGET TO LOSS SHAPE......{target.shape}")
            # TARGET SHAPE IN FORWARD OF LOSS torch.Size([1, 53, 53]) INPUT SHAPE IN FORWARD OF LOSS torch.Size([20, 81, 53, 53])
            # print(f'TARGET SHAPE IN FORWARD OF LOSS {target.shape} INPUT SHAPE IN FORWARD OF LOSS {inputs.shape}')
            # target = target.repeat(20, 1, 1) 
            # .torch.Size([20, 1, 417, 417])
            # inputs = inputs.reshape(1, 20, 53, 53)
            # inputs = inputs.view(1, 20, 53, 53)
            # inputs = inputs.repeat(1, 20, 1, 1)
            inputs = inputs.reshape(1, -1, 53, 53)  
            # print(f'TARGET SHAPE IN FORWARD OF LOSS {target.shape} INPUT SHAPE IN FORWARD OF LOSS {inputs.shape}')
            # target = target.repeat(20, 1, 1)  # Repeat the target tensor along the first dimension (batch size)
            # print(f"TARGET TO LOSS SHAPED HERE......{target.shape}")  # Output: TARGET TO LOSS SHAPE......torch.Size([20, 53, 53])
            # TARGET SHAPE IN FORWARD OF LOSS torch.Size([1, 512, 1024]) INPUT SHAPE IN FORWARD OF LOSS torch.Size([1, 19, 512, 1024])
            loss = self.ce_loss(inputs, target)
            # print(f'loss value....{loss}')

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        # targets = targets_.clone().float()
        # print(f'shape of target......{targets.shape}')
        # print(f'shape of scaled_size...{scaled_size}')
        # Remove singleton dimensions
        # targets = targets.reshape(20, 1, 417, 417)
        # print(f'shape of reshape target...{targets.shape}')
        targets = F.interpolate(targets, size=scaled_size, mode='nearest') 
        return targets.squeeze(1).long()

class FSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss
