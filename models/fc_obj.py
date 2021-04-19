import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as tmodels
import numpy as np
import logging

from utils import utils, aux_data
# from utils import config as cfg


class MLP(nn.Module):
    """Multi-layer perceptron, 1 layers as default. No activation after last fc"""
    def __init__(self, inp_dim, out_dim, hidden_layers=[], batchnorm=True, bias=True)
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
        output = torch.mean(metric)
        return output


class TripletMarginLoss(Distance):
    def __init__(self, margin, metric):
        super(TripletMarginLoss, self).__init__(metric)
        self.triplet_margin = margin


    def forward(self, anchor, positive, negative):
        pos_dist = self.metric_func(anchor, positive)
        neg_dist = self.metric_func(anchor, negative)
        dist_diff = pos_dist - neg_dist + self.triplet_margin
        return torch.maximum(dist_diff, 0)



class Model(nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()

        self.num_attr = dataset.num_attr
        self.num_pair = dataset.num_pair
        
        self.obj_cls_mlp_layer = MLP(
            dataset.feat_dim, dataset.num_obj, 
            hidden_layers=args.fc_cls,
            batchnorm=args.batchnorm)

        if args.loss_class_weight:
            _, obj_loss_weight = aux_data.load_loss_weight(args.data)
            obj_loss_weight = torch.from_numpy(obj_loss_weight).cuda()
            self.obj_bce = nn.CrossEntropyLoss(weight=obj_loss_weight)
        else:
            self.obj_bce = nn.CrossEntropyLoss()



    def forward(self, batch):
        score_obj = self.obj_cls_mlp_layer(batch["pos_feature"])
        prob_obj = F.softmax(score_obj, dim=-1)
        loss = self.obj_bce(score_obj, batch["pos_obj_id"])

        prob_attr = torch.zeros([prob_obj.size(0), self.num_attr]).cuda()

        return prob_attr, prob_obj, {"total": loss}