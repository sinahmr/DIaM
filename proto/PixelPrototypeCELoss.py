from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
from .loss_helper import FSAuxRMILoss, FSCELoss
# from lib.utils.tools.logger import Logger as Log


class PPC(nn.Module, ABC):
    def __init__(self, configer):
        super(PPC, self).__init__()
       
        
        # with open("config/H_48_D_4_proto.json", "r") as f:
        #     self.configer = json.load(f)
        
        # dataset_name = configer["dataset"]
        # num_classes = configer["data"]["num_classes"]
        # ce_weight = configer["loss"]["params"]["ce_weight"]
        self.ignore_label = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, configer):
        super(PPD, self).__init__()

        # with open("config/H_48_D_4_proto.json", "r") as f:
        #     self.configer = json.load(f)

        # dataset_name = configer["dataset"]
        # num_classes = configer["data"]["num_classes"]
        # ce_weight = configer["loss"]["params"]["ce_weight"]

        self.ignore_label = -1
        # if "loss" in configer and "params" in configer["loss"] and "ce_weight" in configer["loss"]["params"]:
        #     self.ignore_label  = torch.FloatTensor(configer["loss"]["params"]["ce_weight"]).cuda()
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()
        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss, self).__init__()
        loss_ppc_weight = 0.01
        loss_ppd_weight = 0.001
        use_rmi = False
        # self.configer = configer

        ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        # Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight

        self.use_rmi = False

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        print("first THE PPD LOSS", preds.size())
        print("first TARGET", target.size())
        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
            
        # concatenated_tensor = torch.cat([preds[0],preds[1]], dim=0) 
        print("HITTING THE PPD LOSS", preds.size())
        aux_out, seg_out = preds
        seg = seg_out
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss
