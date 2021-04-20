import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class MLP(nn.Module):
    """Multi-layer perceptron, 1 layers as default. No activation after last fc"""
    def __init__(self, inp_dim, out_dim, hidden_layers=[], batchnorm=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        last_dim = inp_dim
        for hid_dim in hidden_layers:
            mod.append(nn.Linear(last_dim, hid_dim, bias=bias))
            if batchnorm:
                mod.append(nn.BatchNorm1d(hid_dim))
            mod.append(nn.ReLU(inplace=True))
            last_dim = hid_dim

        mod.append(nn.Linear(last_dim, out_dim, bias=bias))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class Distance(nn.Module):
    def __init__(self, metric):
        super(Distance, self).__init__()
        
        if metric == "L2":
            self.metric_func = lambda x, y: torch.norm(x-y, 2, dim=-1)
        elif metric == "L1":
            self.metric_func = lambda x, y: torch.norm(x-y, 1, dim=-1)
        elif metric == "cos":
            self.metric_func = lambda x, y: 1-F.cosine_similarity(x, y, dim=-1)
        else:
            raise NotImplementedError("Unsupported distance metric: %s"%metric)

    def forward(self, x, y):
        output = self.metric_func(x, y)
        return output


class DistanceLoss(Distance):
    def forward(self, x, y):
        output = self.metric_func(x, y)
        output = torch.mean(output)
        return output


class TripletMarginLoss(Distance):
    def __init__(self, margin, metric):
        super(TripletMarginLoss, self).__init__(metric)
        self.triplet_margin = margin


    def forward(self, anchor, positive, negative):
        pos_dist = self.metric_func(anchor, positive)
        neg_dist = self.metric_func(anchor, negative)
        dist_diff = pos_dist - neg_dist + self.triplet_margin
        output = torch.max(dist_diff, torch.zeros_like(dist_diff).to(dist_diff.device))
        return output.mean()


class CrossEntropyLossWithProb(nn.Module):
    def __init__(self, weight=None, clip_thres=1e-8):
        super(CrossEntropyLossWithProb, self).__init__()
        self.nll = nn.NLLLoss(weight)
        self.clip_thres = clip_thres

    def forward(self, probs, labels):
        probs = probs.clamp_min(self.clip_thres)
        ll = torch.log(probs)
        return self.nll(ll, labels)