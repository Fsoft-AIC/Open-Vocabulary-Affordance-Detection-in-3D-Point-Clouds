import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EstimationLoss(nn.Module):
    def __init__(self, cfg):
        super(EstimationLoss, self).__init__()
        self.weights = torch.from_numpy(np.load(cfg.training_cfg.weights_dir)).cuda().float()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target, weight=self.weights)
        return total_loss
