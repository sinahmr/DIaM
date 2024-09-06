# ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ## Created by: Donny You, RainbowSecret
# ## Microsoft Research
# ## yuyua@microsoft.com
# ## Copyright (c) 2019
# ##
# ## This source code is licensed under the MIT-style license found in the
# ## LICENSE file in the root directory of this source tree 
# ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .rmi_loss import RMILoss
import json

# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer):
        super(FSCELoss, self).__init__()
        self.configer = configer

        with open("config/H_48_D_4_proto.json", "r") as f:
            self.configer = json.load(f)
        # print("CONFIGURE....",self.configure)
        # print("CONFIGURE....",configure)
        weight = None
         # Access and print a specific config value (example)
        if "loss" in self.configer and "params" in self.configer["loss"]:
            ce_weight = self.configer["loss"]["params"].get("ce_weight")
            if ce_weight is not None:
                print("CE weight from config:", ce_weight)
        else:
            print("CE weight not found in config")
        # if "loss" in configer and "params" in configer["loss"] and "ce_weight" in configer["loss"]["params"]:
        #     weight = self.configer["loss"]["params"]['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()
        # if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
        #     weight = self.configer.get('loss', 'params')['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()

        reduction = 'mean'
        if "loss" in self.configer and "params" in self.configer["loss"] and "ce_reduction" in self.configer["loss"]["params"]:
             reduction = self.configer["loss"]["params"]['ce_reduction']
        # if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
        #     reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if "loss" in self.configer and "params" in self.configer["loss"] and "ce_weight" in self.configer["loss"]["params"]:
             ignore_index = self.configer["loss"]["params"]['ce_ignore_index']
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

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
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSAuxRMILoss(nn.Module):
    def __init__(self, configer):
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



