import torch
import torch.nn as nn
import torch.nn.functional as fc_cls
import numpy as np
import logging

from utils import utils, aux_data
from base_models import MLP



class Model(nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()

        self.num_attr = len(dataset.attrs)
        self.num_pair = len(dataset.pairs)
        
        self.obj_cls_mlp_layer = MLP(
            dataset.feat_dim, len(dataset.objs), 
            hidden_layers=args.fc_cls,
            batchnorm=args.batchnorm)

        if args.loss_class_weight:
            _, obj_loss_weight = aux_data.load_loss_weight(args.data)
            obj_loss_weight = torch.tensor(obj_loss_weight, dtype=torch.float32).cuda()
            self.obj_bce = nn.CrossEntropyLoss(weight=obj_loss_weight)
        else:
            self.obj_bce = nn.CrossEntropyLoss()



    def forward(self, batch):
        score_obj = self.obj_cls_mlp_layer(batch["pos_feature"])
        prob_obj = F.softmax(score_obj, dim=-1)
        loss = self.obj_bce(score_obj, batch["pos_obj_id"])

        prob_attr = torch.zeros([prob_obj.size(0), self.num_attr]).cuda()

        return prob_attr, prob_obj, {"total": loss}