import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from utils import utils, aux_data
from .base_models import MLP, BCELossWithLabel



class Model(nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()

        self.num_obj  = len(dataset.objs)
        self.num_attr = len(dataset.attrs)
        
        self.obj_cls_mlp_layer = MLP(
            dataset.feat_dim, self.num_obj, 
            hidden_layers=args.fc_cls,
            batchnorm=args.batchnorm)

        if args.loss_class_weight:
            _, obj_loss_wgt = aux_data.load_loss_weight(args.data)
            obj_loss_wgt = torch.tensor(obj_loss_wgt,  dtype=torch.float32).cuda()
            self.obj_bce = BCELossWithLabel(self.num_obj, weight=obj_loss_wgt)
        else:
            self.obj_bce = BCELossWithLabel(self.num_obj)




    def forward(self, batch):
        score_obj = self.obj_cls_mlp_layer(batch["pos_feature"])
        prob_obj = F.softmax(score_obj, dim=-1)
        loss = self.obj_bce(prob_obj, batch["pos_obj_id"])

        foo_prob_attr = torch.zeros([prob_obj.size(0), self.num_attr]).cuda()
        
        return foo_prob_attr, prob_obj, {"loss_total": loss}