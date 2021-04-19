import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from utils import utils, aux_data
from .base_models import *

class Transformer(nn.Module):
    def __init__(self, attr_emb_dim, args, name):
        super(Transformer, self).__init__()
        self.name = name

        if not args.no_attention:
            self.fc_attention = MLP(attr_emb_dim, args.rep_dim, hidden_layers=args.fc_att, batchnorm=args.batchnorm)
        else:
            self.fc_attention = None

        self.fc_out = MLP(args.rep_dim + attr_emb_dim, args.rep_dim, hidden_layers=args.fc_compress, batchnorm=args.batchnorm)

    def forward(self, rep, v_attr):
        if self.fc_attention:
            attention = self.fc_attention(v_attr)
            attention = torch.sigmoid(attention)
            rep = attention * rep + rep

        hidden = torch.cat((rep, v_attr), dim=1)
        output = self.fc_out(hidden)

        return output

    def string(self):
        return self.name



class MLPClassifier(nn.Module):
    def __init__(self, num_class, args, name):
        super(MLPClassifier, self).__init__()
        self.name = name
        self.mlp = MLP(args.rep_dim, num_class, hidden_layers=args.fc_cls, batchnorm=args.batchnorm)

    def forward(self, emb):
        score = self.mlp(emb)
        prob = F.softmax(score, -1)
        return score, prob

    def string(self):
        return self.name


class Model(nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()

        self.args = args

        self.num_obj  = len(dataset.objs)
        self.num_attr = len(dataset.attrs)

        self.attr_embedder = utils.Embedder(args.wordvec, dataset.attrs, args.data)
        self.emb_dim = self.attr_embedder.emb_dim  # dim of wordvec (attr or obj)

        self.rep_embedder = MLP(dataset.feat_dim, args.rep_dim, hidden_layers=[], batchnorm=args.batchnorm)

        self.CoN = Transformer(self.emb_dim, args, 'CoN')
        self.DeCoN = Transformer(self.emb_dim, args, 'DeCoN')

        self.attr_cls = MLPClassifier(self.num_attr, args, "attr_cls")
        self.obj_cls = MLPClassifier(self.num_obj, args, "obj_cls")

        if args.loss_class_weight:
            attr_loss_wgt, obj_loss_wgt = aux_data.load_loss_weight(args.data)
            attr_loss_wgt= torch.tensor(attr_loss_wgt, dtype=torch.float32).cuda()
            obj_loss_wgt= torch.tensor(obj_loss_wgt, dtype=torch.float32).cuda()
            self.attr_ce = nn.CrossEntropyLoss(weight=attr_loss_wgt)
            self.attr_ce_prob = CrossEntropyLossWithProb(weight=attr_loss_wgt)
            self.obj_ce = nn.CrossEntropyLoss(weight=obj_loss_wgt)
        else:
            self.attr_ce = nn.CrossEntropyLoss()
            self.attr_ce_prob = CrossEntropyLossWithProb()
            self.obj_ce = nn.CrossEntropyLoss()

        self.dist_func = Distance(args.distance_metric)
        self.dist_loss = DistanceLoss(args.distance_metric)
        self.triplet_loss = TripletMarginLoss(args.triplet_margin, args.distance_metric)


    def forward(self, batch, require_loss=True):
        pos_image_feat  = batch["pos_feature"]
        batchsize = pos_image_feat.shape[0]

        pos_img = self.rep_embedder(pos_image_feat)  # (bz,dim)
        

        # obj prediction
        score_pos_obj, prob_pos_obj = self.obj_cls(pos_img)

        # attribute prediction (RMD)
        attr_emb = self.attr_embedder.get_embedding(torch.arange(0, self.num_attr))
        # (#attr, dim_emb), wordvec of all attributes
        tile_attr_emb = utils.tile_on(attr_emb, batchsize, 0)
        # (bz*#attr, dim_emb)
        
        repeat_img_feat = utils.repeat_on(pos_img, self.num_attr, 0)
        # (bz*#attr, dim_rep)
        feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
        feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)

        prob_RMD_plus, prob_RMD_minus = self.RMD_prob(
            feat_plus, feat_minus,
            repeat_img_feat,
            self.args.rmd_metric)

        prob_attr = (prob_RMD_plus+prob_RMD_minus)*0.5
        
        if "obj_pred" in batch:
            prob_obj = batch["obj_pred"]
        else:
            prob_obj = prob_pos_obj


        if require_loss:
            losses = self.compute_loss(batch, pos_img, score_pos_obj, prob_RMD_plus, prob_RMD_minus)
            return prob_attr, prob_obj, losses
        else:
            return prob_attr, prob_obj


    def compute_loss(self, batch, pos_img, score_pos_obj, prob_RMD_plus, prob_RMD_minus):
        losses = {}

        pos_attr_id     = batch["pos_attr_id"]
        pos_obj_id      = batch["pos_obj_id"]
        neg_attr_id     = batch["neg_attr_id"]
        neg_image_feat  = batch["neg_feature"]

        pos_attr_emb = self.attr_embedder.get_embedding(pos_attr_id)
        neg_attr_emb = self.attr_embedder.get_embedding(neg_attr_id)

        neg_img = self.rep_embedder(neg_image_feat)  # (bz,dim)

        # rA = remove positive attribute A
        # aA = add positive attribute A
        # rB = remove negative attribute B
        # aB = add negative attribute B
        pos_aA = self.CoN(pos_img, pos_attr_emb)
        pos_aB = self.CoN(pos_img, neg_attr_emb)
        pos_rA = self.DeCoN(pos_img, pos_attr_emb)
        pos_rB = self.DeCoN(pos_img, neg_attr_emb)


        ########################## classification losses ######################
        # unnecessary to compute cls loss for neg_img

        if self.args.lambda_cls_attr > 0:
            # original image
            score_pos_A, prob_pos_A = self.attr_cls(pos_img)
            losses["loss_cls_attr/pos_a"] = self.attr_ce(score_pos_A, pos_attr_id)

            # after removing pos_attr
            score_pos_rA_A, prob_pos_rA_A = self.attr_cls(pos_rA)
            losses["loss_cls_attr/pos_rA_a"] = self.attr_ce(-score_pos_rA_A, pos_attr_id) # different implementation from TF version

            # rmd
            losses["loss_cls_attr/rmd_plus"] = self.attr_ce_prob(prob_RMD_plus, pos_attr_id)
            losses["loss_cls_attr/rmd_minus"] = self.attr_ce_prob(prob_RMD_minus, pos_attr_id)

            # summary
            losses["loss_cls_attr/total"] = sum([
                losses["loss_cls_attr/pos_a"], 
                losses["loss_cls_attr/pos_rA_a"],
                losses["loss_cls_attr/rmd_plus"], 
                losses["loss_cls_attr/rmd_minus"]
            ])
        else:
            losses["loss_cls_attr/total"] = 0


        if self.args.lambda_cls_obj > 0:
            # original image
            losses["loss_cls_obj/pos_o"] = self.obj_ce(score_pos_obj, pos_obj_id)

            # after removing pos_attr
            score_pos_rA_O, prob_pos_rA_O = self.obj_cls(pos_rA)
            losses["loss_cls_obj/pos_rA_o"] = self.obj_ce(score_pos_rA_O, pos_obj_id)

            # after adding neg_attr
            score_pos_aB_O, prob_pos_aB_O = self.obj_cls(pos_aB)
            losses["loss_cls_obj/pos_aB_o"] = self.obj_ce(score_pos_aB_O, pos_obj_id)


            losses["loss_cls_obj/total"] = sum([
                losses["loss_cls_obj/pos_o"],
                losses["loss_cls_obj/pos_rA_o"],
                losses["loss_cls_obj/pos_aB_o"]
            ])
        else:
            losses["loss_cls_obj/total"] = 0



        ############################# symmetry loss ###########################

        if self.args.lambda_sym > 0:
            losses["loss_sym/pos"] = self.dist_loss(pos_aA, pos_img)
            losses["loss_sym/neg"] = self.dist_loss(pos_rB, pos_img)

            losses["loss_sym/total"] = (losses["loss_sym/pos"] + losses["loss_sym/neg"])
        else:
            losses["loss_sym/total"] = 0
            

        ############################## axiom losses ###########################
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0

            # closure
            if self.args.remove_clo:
                losses["loss_axiom/clo"] = 0
            else:
                pos_aA_rA = self.DeCoN(pos_aA, pos_attr_emb)
                pos_rB_aB = self.CoN(pos_rB, neg_attr_emb)
                losses["loss_axiom/clo"] = self.dist_loss(pos_aA_rA, pos_rA) + \
                           self.dist_loss(pos_rB_aB, pos_aB)

            # invertibility
            if self.args.remove_inv:
                losses["loss_axiom/inv"] = 0
            else:
                pos_rA_aA = self.CoN(pos_rA, pos_attr_emb)
                pos_aB_rB = self.DeCoN(pos_aB, neg_attr_emb)
                losses["loss_axiom/inv"] = self.dist_loss(pos_rA_aA, pos_img) + \
                           self.dist_loss(pos_aB_rB, pos_img)

            # Commutativity
            if self.args.remove_com:
                losses["loss_axiom/com"] = 0
            else:
                pos_aA_rB = self.DeCoN(pos_aA, neg_attr_emb)
                pos_rB_aA = self.CoN(pos_rB, pos_attr_emb)
                losses["loss_axiom/com"] = self.dist_loss(pos_aA_rB, pos_rB_aA)

            losses["loss_axiom/total"] = (
                    losses["loss_axiom/clo"] + losses["loss_axiom/inv"] + losses["loss_axiom/com"])

        else:
            losses["loss_axiom/total"] = 0
                    

        ############################# triplet loss ############################

        if self.args.lambda_trip > 0:
            losses["loss_trip/pos"] = self.triplet_loss(pos_img, pos_aA, pos_rA)
            losses["loss_trip/neg"] = self.triplet_loss(pos_img, pos_rB, pos_aB)
            losses["loss_trip/total"] = (losses["loss_trip/pos"] + losses["loss_trip/neg"])

        else:
            losses["loss_trip/total"] = 0

        ############################### summary ###############################
        losses["loss_total"] = (
            self.args.lambda_cls_attr * losses["loss_cls_attr/total"] +
            self.args.lambda_cls_obj * losses["loss_cls_obj/total"] +
            self.args.lambda_sym * losses["loss_sym/total"] +
            self.args.lambda_axiom * losses["loss_axiom/total"] + 
            self.args.lambda_trip * losses["loss_trip/total"]
        )

        return losses




    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat, rmd_metric):
        """return attribute classification probability with our RMD"""
        # feat_plus, feat_minus:  shape=(bz, #attr, dim_emb)
        # d_plus: distance between feature before&after CoN
        # d_minus: distance between feature before&after DecoN

        d_plus = self.dist_func(feat_plus, repeat_img_feat)
        d_minus = self.dist_func(feat_minus, repeat_img_feat)
        d_plus = torch.reshape(d_plus, [-1, self.num_attr])  # bz, #attr
        d_minus = torch.reshape(d_minus, [-1, self.num_attr])  # bz, #attr

        if rmd_metric == 'softmax':
            p_plus = F.softmax(-d_plus, 1)  # (bz, #attr), smaller = better
            p_minus = F.softmax(d_minus, 1)  # (bz, #attr), larger = better
            return p_plus, p_minus

        elif rmd_metric == 'rmd':
            # d_plus_comp = torch.from_numpy(self.dset.comp_gamma['b']).to(self.device) * d_plus
            # d_minus_comp = torch.from_numpy(self.dset.comp_gamma['a']).to(self.device) * d_minus
            # d_plus_attr = torch.from_numpy(self.dset.attr_gamma['b']).to(self.device) * d_plus
            # d_minus_attr = torch.from_numpy(self.dset.attr_gamma['a']).to(self.device) * d_minus
            d_plus_comp = d_plus
            d_minus_comp = d_minus
            d_plus_attr = d_plus
            d_minus_attr = d_minus

            p_comp = F.softmax(d_minus_comp - d_plus_comp, dim=1)
            p_attr = F.softmax(d_minus_attr - d_plus_attr, dim=1)

            return p_comp, p_attr
