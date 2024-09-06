import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import yaml
import json
from .loss_helper import FSAuxRMILoss, FSCELoss

def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        # print(f'Q shape is .....{Q.shape}')
        Q = Q.T
        # B = Q.shape[1] * Q.shape[2]
        B = Q.shape[1] * Q.shape[0]
        # print(f'print set....{B}')
        K = Q.shape[0]
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(nmb_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B  # the colomns must sum to 1
        Q = Q.T
    return Q, torch.argmax(Q, dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, drop_rate=0.1):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Linear(dim_in, proj_dim)
        self.norm = nn.BatchNorm1d(proj_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class PixelPrototypeClassifier(nn.Module):
    def __init__(self, configer: dict, backbone, feature_dim, num_prototypes):
        super(PixelPrototypeClassifier, self).__init__()
        self.configer = configer
        self.gamma = self.configer['protoseg']['gamma']
        self.num_prototype = self.configer['protoseg']['num_prototype']
        self.use_prototype = self.configer['protoseg']['use_prototype']
        self.update_prototype = self.configer['protoseg']['update_prototype']
        self.pretrain_prototype = self.configer['protoseg']['pretrain_prototype']
        self.num_classes = self.configer['data']['num_classes']
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Prototype layer
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, num_prototypes, self.feature_dim),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(self.feature_dim, self.feature_dim)
        self.feat_norm = nn.LayerNorm(self.feature_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, features, gt_seg):
        # Flatten features
        _c = rearrange(features, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)  # Assuming l2_normalize is defined elsewhere

        # Normalize prototypes
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # Cosine similarity
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=features.shape[0], h=features.shape[2])

        if self.use_prototype and gt_seg is not None:
            gt_seg = F.interpolate(gt_seg.float(), size=features.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning_step(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        return out_seg

    def prototype_learning_step(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue
            # Assuming distributed_sinkhorn is defined elsewhere
            q, indexs = distributed_sinkhorn(init_q, nmb_iters=3)
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile
            f = m_q.transpose(0, 1) @ c_q
            n = torch.sum(m_q, dim=0)
            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma)
                protos[k, n != 0, :] = new_value
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False) 
        return proto_logits, proto_target

    def forward(self, x, gt_semantic_seg=None, pretrain_prototype=False):
        features = self.backbone(x)[-1]
        features = self.proj_head(features)
        if pretrain_prototype is False and self.use_prototype and gt_semantic_seg is not None:
            output = self.prototype_learning(features, gt_semantic_seg)
        else:
            output = self.prototype_learning(features, None)
        return output

class PPC(nn.Module):
    def __init__(self, ignore_label=-1):
        super(PPC, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        return loss_ppc

class PPD(nn.Module):
    def __init__(self, ignore_label=-1):
        super(PPD, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()
        return loss_ppd

class PixelPrototypeCELoss(nn.Module):
    def __init__(self, configer=None, use_rmi=False):
        super(PixelPrototypeCELoss, self).__init__()
        self.configer = configer
        ignore_index = -1
        if 'loss' in self.configer and 'params' in self.configer['loss'] and 'ce_ignore_index' in self.configer['loss']['params']:
            ignore_index = self.configer['loss']['params']['ce_ignore_index']

        self.loss_ppc_weight = self.configer['protoseg']['loss_ppc_weight']
        self.loss_ppd_weight = self.configer['protoseg']['loss_ppd_weight']

        self.use_rmi = use_rmi

        # Replace with your appropriate segmentation loss
        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(ignore_label=ignore_index) 
        self.ppd_criterion = PPD(ignore_label=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if isinstance(preds, dict):
            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss_seg = self.seg_criterion(pred, target)
            return loss_seg + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        # Assuming preds is the segmentation output directly
        pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss_seg = self.seg_criterion(pred, target)
        return loss_seg 

def l2_normalize(x):
    return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10)

def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update