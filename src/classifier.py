from typing import Tuple

import torch
import torch.nn.functional as F

from src.util import compute_wce
from .util import to_one_hot
from proto.PixelPrototypeCELoss import PixelPrototypeCELoss

class Classifier(object):
    def __init__(self, args, base_weight, base_bias, n_tasks):
        self.num_base_classes_and_bg = base_weight.size(-1)
        self.num_novel_classes = args.num_classes_val
        self.num_classes = self.num_base_classes_and_bg + self.num_novel_classes
        self.n_tasks = n_tasks

        self.snapshot_weight = base_weight.squeeze(0).squeeze(0).clone()  # Snapshot of the model right after training, frozen
        self.snapshot_bias = base_bias.clone()
        self.base_weight = base_weight.squeeze(0).repeat(self.n_tasks, 1, 1)  # [n_tasks, c, num_base_classes_and_bg]
        self.base_bias = base_bias.unsqueeze(0).repeat(self.n_tasks, 1)  # [n_tasks, num_base_classes_and_bg]

        self.novel_weight, self.novel_bias = None, None
        self.pi, self.true_pi = None, None

        self.fine_tune_base_classifier = args.fine_tune_base_classifier
        self.lr = args.cls_lr
        self.adapt_iter = args.adapt_iter
        self.weights = args.weights
        self.pi_estimation_strategy = args.pi_estimation_strategy
        self.pi_update_at = args.pi_update_at

    @staticmethod
    def _valid_mean(t, valid_pixels, dim):
        s = (valid_pixels * t).sum(dim=dim)
        return s / (valid_pixels.sum(dim=dim) + 1e-10)

    def init_prototypes(self, features_s: torch.tensor, gt_s: torch.tensor) -> None:
        """
        inputs:
            features_s : shape [num_novel_classes, shot, c, h, w]
            gt_s : shape [num_novel_classes, shot, H, W]
        """
        # Downsample support masks
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_novel_classes, shot, 1, h, w]

        # Computing prototypes
        self.novel_weight = torch.zeros((features_s.size(2), self.num_novel_classes), device=features_s.device)
        for cls in range(self.num_base_classes_and_bg, self.num_classes):
            novel_mask = (ds_gt_s == cls)
            novel_prototype = self._valid_mean(features_s, novel_mask, (0, 1, 3, 4))  # [c,]
            self.novel_weight[:, cls - self.num_base_classes_and_bg] = novel_prototype

        self.novel_weight /= self.novel_weight.norm(dim=0).unsqueeze(0) + 1e-10
        assert torch.isnan(self.novel_weight).sum() == 0, self.novel_weight
        self.novel_bias = torch.zeros((self.num_novel_classes,), device=features_s.device)

        # Copy prototypes for each task
        self.novel_weight = self.novel_weight.unsqueeze(0).repeat(self.n_tasks, 1, 1)
        self.novel_bias = self.novel_bias.unsqueeze(0).repeat(self.n_tasks, 1)

    def get_logits(self, features: torch.tensor) -> torch.tensor:
        """
        Computes logits for given features

        inputs:
            features : shape [1 or batch_size_val, num_novel_classes * shot or 1, c, h, w]

        returns :
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        equation = 'bochw,bcC->boChw'  # 'o' is n_novel_classes * shot for support and is 1 for query

        novel_logits = torch.einsum(equation, features, self.novel_weight)
        base_logits = torch.einsum(equation, features, self.base_weight)
        novel_logits += self.novel_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        base_logits += self.base_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        logits = torch.concat([base_logits, novel_logits], dim=2)
        return logits

    @staticmethod
    def get_probas(logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]

        returns :
            probas : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        return torch.softmax(logits, dim=2)

    def get_base_snapshot_probas(self, features: torch.tensor) -> torch.tensor:
        """
        Computes probability maps for given query features, using the snapshot of the base model right after the
        training. It only computes values for base classes.

        inputs:
            features : shape [batch_size_val, 1, c, h, w]

        returns :
            probas : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
        """
        logits = torch.einsum('bochw,cC->boChw', features, self.snapshot_weight) + self.snapshot_bias.view(1, 1, -1, 1, 1)
        return torch.softmax(logits, dim=2)

    def self_estimate_pi(self, features_q: torch.tensor, unsqueezed_valid_pixels_q: torch.tensor) -> torch.tensor:
        """
        Estimates pi using model's prototypes

        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            unsqueezed_valid_pixels_q : shape [batch_size_val, 1, 1, h, w]

        returns :
            pi : shape [batch_size_val, num_classes]
        """
        logits_q = self.get_logits(features_q)
        probas = torch.softmax(logits_q, dim=2).detach()
        return self._valid_mean(probas, unsqueezed_valid_pixels_q, (1, 3, 4))

    def image_level_supervision_pi(self, features_q: torch.tensor,
                                   unsqueezed_valid_pixels_q: torch.tensor) -> torch.tensor:
        """
        Estimates pi using model's prototypes and information about whether each class is present in a query image.

        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            unsqueezed_valid_pixels_q : shape [batch_size_val, 1, 1, h, w]

        returns :
            pi : shape [batch_size_val, num_classes]
        """
        logits_q = self.get_logits(features_q)
        absent_indices = torch.where(self.true_pi == 0)
        logits_q[absent_indices[0], :, absent_indices[1], :, :] = -torch.inf
        probas = torch.softmax(logits_q, dim=2).detach()
        return self._valid_mean(probas, unsqueezed_valid_pixels_q, (1, 3, 4))

    def compute_pi(self, features_q: torch.tensor, valid_pixels_q: torch.tensor,
                   gt_q: torch.tensor = None) -> torch.tensor:
        """
        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            valid_pixels_q : shape [batch_size_val, 1, h, w]
            gt_q : shape [batch_size_val, 1, H, W]
        """
        valid_pixels_q = F.interpolate(valid_pixels_q.float(), size=features_q.size()[-2:], mode='nearest').long()
        valid_pixels_q = valid_pixels_q.unsqueeze(2)

        if gt_q is not None:
            ds_gt_q = F.interpolate(gt_q.float(), size=features_q.size()[-2:], mode='nearest').long()
            one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [batch_size_val, shot, num_classes, h, w]
            self.true_pi = self._valid_mean(one_hot_gt_q, valid_pixels_q, (1, 3, 4))

        if self.pi_estimation_strategy == 'upperbound':
            self.pi = self.true_pi
        elif self.pi_estimation_strategy == 'self':
            self.pi = self.self_estimate_pi(features_q, valid_pixels_q)
        elif self.pi_estimation_strategy == 'imglvl':
            self.pi = self.image_level_supervision_pi(features_q, valid_pixels_q)
        elif self.pi_estimation_strategy == 'uniform':
            pi = 1 / self.num_classes
            self.pi = torch.full_like(self.true_pi, pi)  # [batch_size_val, num_classes]
        else:
            raise ValueError('pi_estimation_strategy is not implemented')

    def distillation_loss(self, curr_p: torch.tensor, snapshot_p: torch.tensor, valid_pixels: torch.tensor,
                          reduction: str = 'mean') -> torch.tensor:
        """
        inputs:
            curr_p : shape [batch_size_val, 1, num_classes, h, w]
            snapshot_p : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
            valid_pixels : shape [batch_size_val, 1, h, w]

        returns:
             kl : Distillation loss for the query
        """
        adjusted_curr_p = curr_p.clone()[:, :, :self.num_base_classes_and_bg, ...]
        adjusted_curr_p[:, :, 0, ...] += curr_p[:, :, self.num_base_classes_and_bg:, ...].sum(dim=2)
        kl = (adjusted_curr_p * torch.log(1e-10 + adjusted_curr_p / (1e-10 + snapshot_p))).sum(dim=2)
        kl = self._valid_mean(kl, valid_pixels, (1, 2, 3))
        if reduction == 'sum':
            kl = kl.sum(0)
        elif reduction == 'mean':
            kl = kl.mean(0)
        return kl

    def get_entropies(self, valid_pixels: torch.tensor, probas: torch.tensor,
                      reduction: str = 'mean') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        inputs:
            valid_pixels: shape [batch_size_val, 1, h, w]
            probas : shape [batch_size_val, 1, num_classes, h, w]

        returns:
            d_kl : Classes proportion kl
            entropy : Entropy of predictions
            marginal : Current marginal distribution over labels [batch_size_val, num_classes]
        """
        entropy = - (probas * torch.log(probas + 1e-10)).sum(2)
        entropy = self._valid_mean(entropy, valid_pixels, (1, 2, 3))
        marginal = self._valid_mean(probas, valid_pixels.unsqueeze(2), (1, 3, 4))

        d_kl = (marginal * torch.log(1e-10 + marginal / (self.pi + 1e-10))).sum(1)

        if reduction == 'sum':
            entropy = entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(entropy), entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            entropy = entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, entropy, marginal

    def get_ce(self, probas: torch.tensor, valid_pixels: torch.tensor, one_hot_gt: torch.tensor,
               reduction: str = 'mean') -> torch.tensor:
        """
        inputs:
            probas : shape [batch_size_val, num_novel_classes * shot, c, h, w]
            valid_pixels : shape [1, num_novel_classes * shot, h, w]
            one_hot_gt: shape [1, num_novel_classes * shot, num_classes, h, w]

        returns:
             ce : Cross-Entropy between one_hot_gt and probas
        """
        probas = probas.clone()
        probas[:, :, 0, ...] += probas[:, :, 1:self.num_base_classes_and_bg, ...].sum(dim=2)
        probas[:, :, 1:self.num_base_classes_and_bg, ...] = 0.

        ce = - (one_hot_gt * torch.log(probas + 1e-10))
        ce = (ce * compute_wce(one_hot_gt, self.num_novel_classes)).sum(2)
        ce = self._valid_mean(ce, valid_pixels, (1, 2, 3))  # [batch_size_val,]

        if reduction == 'sum':
            ce = ce.sum(0)
        elif reduction == 'mean':
            ce = ce.mean(0)
        return ce

    def optimize(self, features_s: torch.tensor, features_q: torch.tensor, gt_s: torch.tensor,
                 valid_pixels_q: torch.tensor) -> torch.tensor:
        """
        DIaM inference optimization

        inputs:
            features_s : shape [num_novel_classes, shot, c, h, w]
            features_q : shape [batch_size_val, 1, c, h, w]
            gt_s : shape [num_novel_classes, shot, h, w]
            valid_pixels_q : shape [batch_size_val, 1, h, w]
        """
        l1, l2, l3, l4 = self.weights
        # //added section
        criterion = PixelPrototypeCELoss() 
        params = [self.novel_weight, self.novel_bias]
        if self.fine_tune_base_classifier:
            params.extend([self.base_weight, self.base_bias])
        for m in params:
            m.requires_grad_()
        optimizer = torch.optim.SGD(params, lr=self.lr)

        # Flatten the dimensions of different novel classes and shots
        features_s = features_s.flatten(0, 1).unsqueeze(0)
        gt_s = gt_s.flatten(0, 1).unsqueeze(0)

        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()
        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [1, num_novel_classes * shot, num_classes, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()
        valid_pixels_q = F.interpolate(valid_pixels_q.float(), size=features_q.size()[-2:], mode='nearest').long()

        for iteration in range(self.adapt_iter):
            # Create dictionary with required outputs
            # preds = {"seg": seg_output,  "logits": logits_q,"target": gt_q }
            logits_s, logits_q = self.get_logits(features_s), self.get_logits(features_q)
            proba_s, proba_q = self.get_probas(logits_s), self.get_probas(logits_q)
            preds = {"seg": logits_s,  "logits": logits_q,"target": features_q }
            loss_ppc = criterion(preds, features_q) 
            snapshot_proba_q = self.get_base_snapshot_probas(features_q)
            distillation = self.distillation_loss(proba_q, snapshot_proba_q, valid_pixels_q, reduction='none')
            d_kl, entropy, marginal = self.get_entropies(valid_pixels_q, proba_q, reduction='none')
            ce = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none')
            loss = l1 * ce + l2 * d_kl + l3 * entropy + l4 * distillation

            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()

            # Update pi
            if (iteration + 1) in self.pi_update_at and (self.pi_estimation_strategy == 'self') and (l2 != 0):
                self.compute_pi(features_q, valid_pixels_q)
