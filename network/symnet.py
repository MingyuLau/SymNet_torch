from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, logging, torch, json
import os.path as osp
from collections import OrderedDict

from network.base_network import *
# from base_network import *
from utils import utils
from utils import config as cfg

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class NetWork(BaseNetwork):
    # root_logger = logging.getLogger("network %s" % __file__.split('/')[-1])

    def __init__(self, dataloader, args, feat_dim, device=torch.device('cpu')):
        super().__init__(dataloader, args, feat_dim, device)

        self.dset = self.dataloader.dataset
        self.attr_embedder = utils.Embedder(args.wordvec, self.dset.attrs, args.data, self.device)
        self.emb_dim = self.attr_embedder.emb_dim  # dim of wordvec (attr or obj)

        self.rep_embedder = MLP(feat_dim, args.rep_dim, args, is_training=True,
                                name="embedder", hidden_layers=[])

        emb_feat = None
        attr_emb_feat = None
        for batch_ind, blobs in enumerate(dataloader):
            pos_attr_id = torch.from_numpy(blobs[1])
            pos_image_feat = torch.from_numpy(blobs[4])

            pos_attr_emb = self.attr_embedder.get_embedding(pos_attr_id)
            pos_img = self.rep_embedder(pos_image_feat)  # (bz,dim)

            emb_feat = pos_img.shape[1]
            attr_emb_feat = pos_attr_emb.shape[1]
            break

        self.CoN = NetWork.Transformer(attr_emb_feat, args, True, name='CoN')
        self.DeCoN = NetWork.Transformer(attr_emb_feat, args, True, name='DeCoN')

        self.attr_cls = NetWork.Classifier(emb_feat, self.num_attr, args
                                           , is_training=True, name="attr_cls")
        self.obj_cls = NetWork.Classifier(emb_feat, self.num_obj, args,
                                          is_training=True, name="obj_cls")

    class Transformer(nn.Module):
        def __init__(self, attr_emb_feat, args, is_training, name):
            super(NetWork.Transformer, self).__init__()
            self.v_attr_emb_feat = attr_emb_feat
            self.is_training = is_training
            self.name = name
            self.args = args

            self.fc_attention = MLP(attr_emb_feat, self.args.rep_dim, args, is_training,
                                    name='fc_attention', hidden_layers=self.args.fc_att)
            self.fc_out = MLP(args.rep_dim + attr_emb_feat, self.args.rep_dim, args, is_training,
                              name='fc_out', hidden_layers=self.args.fc_compress)

        def forward(self, rep, v_attr):
            if not self.args.no_attention:
                attention = self.fc_attention(v_attr)
                attention = torch.sigmoid(attention)
                rep = attention * rep + rep

            hidden = torch.cat((rep, v_attr), dim=1)
            output = self.fc_out(hidden)

            return output

        def string(self):
            return self.name

    class Classifier(nn.Module):
        def __init__(self, emb_feat, num_attr, args, is_training, name='classifier'):
            super(NetWork.Classifier, self).__init__()
            self.emb_feat = emb_feat
            self.is_training = is_training
            self.name = name

            self.mlp = MLP(emb_feat, num_attr, args, is_training,
                           "cls", hidden_layers=args.fc_cls)

        def forward(self, emb):
            score_A = self.mlp(emb)
            prob_A = F.softmax(score_A, 1)

            return score_A, prob_A

        def string(self):
            return self.name

    def forward(self, blobs):
        train_summary_op = OrderedDict()

        pos_attr_id = torch.from_numpy(blobs[1]).to(self.device)
        pos_obj_id = torch.from_numpy(blobs[2]).to(self.device)
        pos_image_feat = torch.from_numpy(blobs[4]).to(self.device)
        neg_attr_id = torch.from_numpy(blobs[6]).to(self.device)
        neg_image_feat = torch.from_numpy(blobs[9]).to(self.device)

        train_summary_op['sum/pos_image_feat'] = torch.sum(pos_image_feat)
        train_summary_op['mean/pos_image_feat'] = torch.mean(pos_image_feat)

        total_losses = []

        batchsize = pos_image_feat.shape[0]

        pos_attr_emb = self.attr_embedder.get_embedding(pos_attr_id)
        neg_attr_emb = self.attr_embedder.get_embedding(neg_attr_id)



        pos_img = self.rep_embedder(pos_image_feat)  # (bz,dim)
        neg_img = self.rep_embedder(neg_image_feat)  # (bz,dim)

        train_summary_op['sum/pos_img'] = torch.sum(pos_img)
        train_summary_op['mean/pos_img'] = torch.mean(pos_img)

        # rA = remove positive attribute A
        # aA = add positive attribute A
        # rB = remove negative attribute B
        # aB = add negative attribute B

        pos_aA = self.CoN(pos_img, pos_attr_emb)
        pos_aB = self.CoN(pos_img, neg_attr_emb)
        pos_rA = self.DeCoN(pos_img, pos_attr_emb)
        pos_rB = self.DeCoN(pos_img, neg_attr_emb)

        attr_emb = self.attr_embedder.get_embedding(np.arange(self.num_attr))
        # (#attr, dim_emb), wordvec of all attributes
        tile_attr_emb = utils.tile_tensor(attr_emb, 0, batchsize)
        # (bz*#attr, dim_emb)

        ########################## classification losses ######################
        # unnecessary to compute cls loss for neg_img

        if self.args.lambda_cls_attr > 0:
            # original image
            _, prob_pos_A = self.attr_cls(pos_img)
            train_summary_op['sum/prob_pos_A'] = torch.sum(torch.square(prob_pos_A))
            train_summary_op['mean/prob_pos_A'] = torch.mean(torch.square(prob_pos_A))
            loss_cls_pos_a = F.cross_entropy(prob_pos_A, pos_attr_id, weight=self.attr_weight)
            # TODO: focal loss not implemented

            # after removing pos_attr
            _, prob_pos_rA_A = self.attr_cls(pos_rA)
            loss_cls_pos_rA_a = F.cross_entropy(prob_pos_rA_A, pos_attr_id, weight=self.attr_weight)
            # TODO: focal loss not implemented

            # loss_cls_pos_rA_a = cross_entropy(
            #     prob_pos_rA_A, pos_attr_id, self.num_attr,
            #     target=0, weight=self.attr_weight, focal_loss=self.args.focal_loss)

            # rmd
            repeat_img_feat = utils.repeat_tensor(pos_img, 0, self.num_attr)
            # (bz*#attr, dim_rep)
            feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
            feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)

            prob_RMD_plus, prob_RMD_minus = self.RMD_prob(feat_plus, feat_minus,
                                                          repeat_img_feat, is_training=True,
                                                          metric=self.args.rmd_metric)

            loss_cls_rmd_plus = F.cross_entropy(prob_RMD_plus, pos_attr_id, weight=self.attr_weight)
            loss_cls_rmd_minus = F.cross_entropy(prob_RMD_minus, pos_attr_id, weight=self.attr_weight)
            # TODO: focal loss not implemented

            # loss_cls_rmd_plus = cross_entropy(
            #     prob_RMD_plus, self.pos_attr_id, self.num_attr,
            #     weight=self.attr_weight, focal_loss=self.args.focal_loss)
            # loss_cls_rmd_minus = cross_entropy(
            #     prob_RMD_minus, self.pos_attr_id, self.num_attr,
            #     weight=self.attr_weight, focal_loss=self.args.focal_loss)

            loss_cls_attr = self.args.lambda_cls_attr * sum([
                loss_cls_pos_a, loss_cls_pos_rA_a,
                loss_cls_rmd_plus, loss_cls_rmd_minus
            ])

            total_losses.append(loss_cls_attr)
            train_summary_op['loss/loss_cls_attr'] = loss_cls_attr
            train_summary_op['debug/loss_cls_pos_a'] = loss_cls_pos_a
            train_summary_op['debug/loss_cls_pos_rA_a'] = loss_cls_pos_rA_a
            train_summary_op['debug/loss_cls_rmd_plus'] = loss_cls_rmd_plus
            train_summary_op['debug/loss_cls_rmd_minus'] = loss_cls_rmd_minus

        if self.args.lambda_cls_obj > 0:
            # original image
            _, prob_pos_O = self.obj_cls(pos_img)
            loss_cls_pos_o = F.cross_entropy(prob_pos_O, pos_obj_id, weight=self.obj_weight)
            # loss_cls_pos_o = cross_entropy(
            #     prob_pos_O, pos_obj_id, self.num_obj,
            #     weight=self.obj_weight, focal_loss=self.args.focal_loss)

            # after removing pos_attr
            _, prob_pos_rA_O = self.obj_cls(pos_rA)
            loss_cls_pos_rA_o = F.cross_entropy(prob_pos_rA_O, pos_obj_id, weight=self.obj_weight)
            # loss_cls_pos_rA_o = cross_entropy(
            #     prob_pos_rA_O, pos_obj_id, self.num_obj,
            #     weight=self.obj_weight, focal_loss=self.args.focal_loss)

            # after adding neg_attr
            _, prob_pos_aB_O = self.obj_cls(pos_aB)
            loss_cls_pos_aB_o = F.cross_entropy(prob_pos_aB_O, pos_obj_id, weight=self.obj_weight)
            # loss_cls_pos_aB_o = cross_entropy(
            #     prob_pos_aB_O, pos_obj_id, self.num_obj,
            #     weight=self.obj_weight, focal_loss=self.args.focal_loss)

            loss_cls_obj = self.args.lambda_cls_obj * sum([
                loss_cls_pos_o,
                loss_cls_pos_rA_o,
                loss_cls_pos_aB_o
            ])

            total_losses.append(loss_cls_obj)
            train_summary_op['loss/loss_cls_obj'] = loss_cls_obj

        ############################# symmetry loss ###########################

        if self.args.lambda_sym > 0:
            loss_sym_pos = self.MSELoss(pos_aA, pos_img)
            loss_sym_neg = self.MSELoss(pos_rB, pos_img)

            loss_sym = self.args.lambda_sym * (loss_sym_pos + loss_sym_neg)
            total_losses.append(loss_sym)
            train_summary_op['loss/loss_sym'] = loss_sym

        ############################## axiom losses ###########################
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0

            # closure
            if not self.args.remove_clo:
                pos_aA_rA = self.DeCoN(pos_aA, pos_attr_emb)
                pos_rB_aB = self.CoN(pos_rB, neg_attr_emb)
                loss_clo = self.MSELoss(pos_aA_rA, pos_rA) + \
                           self.MSELoss(pos_rB_aB, pos_aB)

            # invertibility
            if not self.args.remove_inv:
                pos_rA_aA = self.CoN(pos_rA, pos_attr_emb)
                pos_aB_rB = self.DeCoN(pos_aB, neg_attr_emb)
                loss_inv = self.MSELoss(pos_rA_aA, pos_img) + \
                           self.MSELoss(pos_aB_rB, pos_img)

            # Commutativity
            if not self.args.remove_com:
                pos_aA_rB = self.DeCoN(pos_aA, neg_attr_emb)
                pos_rB_aA = self.CoN(pos_rB, pos_attr_emb)
                loss_com = self.MSELoss(pos_aA_rB, pos_rB_aA)

            loss_axiom = self.args.lambda_axiom * (
                    loss_clo + loss_inv + loss_com)
            total_losses.append(loss_axiom)

            train_summary_op['loss/loss_axiom'] = loss_axiom
            train_summary_op['loss/loss_clo'] = loss_clo
            train_summary_op['loss/loss_inv'] = loss_inv
            train_summary_op['loss/loss_com'] = loss_com

        ############################# triplet loss ############################

        if self.args.lambda_trip > 0:
            pos_triplet = torch.mean(
                self.triplet_margin_loss(pos_img, pos_aA, pos_rA))
            neg_triplet = torch.mean(
                self.triplet_margin_loss(pos_img, pos_rB, pos_aB))

            loss_triplet = self.args.lambda_trip * (pos_triplet + neg_triplet)
            total_losses.append(loss_triplet)

            train_summary_op['loss/loss_triplet'] = loss_triplet

        ############################### summary ###############################
        loss = sum(total_losses)

        train_summary_op['loss/loss_total'] = loss

        return loss, train_summary_op

    def test_step(self, blobs):
        dset = self.dataloader.dataset
        test_att = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
        test_obj = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])

        pos_image_feat = torch.from_numpy(blobs[4]).to(self.device)
        test_attr_id = torch.from_numpy(test_att).to(self.device)
        test_obj_id = torch.from_numpy(test_obj).to(self.device)
        if self.args.obj_pred is not None:
            pos_obj_prediction = torch.from_numpy(blobs[-1]).to(self.device)
        batchsize = pos_image_feat.shape[0]

        ################################ testing ##############################
        with torch.no_grad():
            pos_img = self.rep_embedder(pos_image_feat)
        repeat_img_feat = utils.repeat_tensor(pos_img, 0, self.num_attr)
        # (bz*#attr, dim_rep)

        attr_emb = self.attr_embedder.get_embedding(np.arange(self.num_attr))
        # (#attr, dim_emb), wordvec of all attributes
        tile_attr_emb = utils.tile_tensor(attr_emb, 0, batchsize)
        # (bz*#attr, dim_emb)

        feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
        feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)

        prob_A_rmd, prob_A_attr = self.RMD_prob(feat_plus, feat_minus,
                                                repeat_img_feat, is_training=False, metric='rmd')

        with torch.no_grad():
            _, prob_A_fc = self.attr_cls(pos_img)
            _, prob_O_fc = self.obj_cls(pos_img)

        if self.args.obj_pred is None:
            prob_O = prob_O_fc
        else:
            prob_O = pos_obj_prediction

        test_a_onehot = one_hot(test_attr_id, depth=self.num_attr, device=self.device, axis=1)
        test_o_onehot = one_hot(test_obj_id, depth=self.num_obj, device=self.device, axis=1)
        # (n_p, n_o)
        prob_P_rmd = torch.mul(
            torch.matmul(prob_A_rmd, torch.transpose(test_a_onehot, 0, 1)),
            torch.matmul(prob_O, torch.transpose(test_o_onehot, 0, 1))
        )

        score_res = dict([
            ("score_rmd", [prob_P_rmd, prob_A_attr, prob_O]),
        ])

        for key in score_res.keys():
            score_res[key][0] = {
                (a, o): score_res[key][0][:, i]
                for i, (a, o) in enumerate(zip(test_att, test_obj))
            }

        return score_res

    def distance_metric(self, a, b):
        if self.args.distance_metric == 'L2':
            return torch.norm(a - b, dim=-1)
        elif self.args.distance_metric == 'L1':
            return torch.norm(a - b, dim=-1, p=1)
        elif self.args.distance_metric == 'cos':
            return torch.sum(
                F.normalize(a, p=2, dim=-1) *
                F.normalize(b, p=2, dim=-1)
                , dim=-1)
        else:
            raise NotImplementedError("Unsupported distance metric: %s" + \
                                      self.args.distance_metric)

    def MSELoss(self, a, b):
        return torch.mean(self.distance_metric(a, b))

        # def L2distance(self, a, b):
        #    return self.distance_metric(a, b)

    def triplet_margin_loss(self, anchor, positive, negative):
        dist = self.distance_metric(anchor, positive) \
               - self.distance_metric(anchor, negative) + self.args.triplet_margin
        return torch.max(dist, torch.zeros(dist.shape).to(self.device))

    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat, is_training, metric):
        """return attribute classification probability with our RMD"""
        # feat_plus, feat_minus:  shape=(bz, #attr, dim_emb)
        # d_plus: distance between feature before&after CoN
        # d_minus: distance between feature before&after DecoN

        d_plus = self.distance_metric(feat_plus, repeat_img_feat)
        d_minus = self.distance_metric(feat_minus, repeat_img_feat)
        d_plus = torch.reshape(d_plus, [-1, self.num_attr])  # bz, #attr
        d_minus = torch.reshape(d_minus, [-1, self.num_attr])  # bz, #attr

        if metric == 'softmax':
            p_plus = F.softmax(-d_plus, 1)  # (bz, #attr), smaller = better
            p_minus = F.softmax(d_minus, 1)  # (bz, #attr), larger = better
            return p_plus, p_minus

        elif metric == 'rmd':
            d_plus_comp = torch.from_numpy(self.dset.comp_gamma['b']).to(self.device) * d_plus
            d_minus_comp = torch.from_numpy(self.dset.comp_gamma['a']).to(self.device) * d_minus
            d_plus_attr = torch.from_numpy(self.dset.attr_gamma['b']).to(self.device) * d_plus
            d_minus_attr = torch.from_numpy(self.dset.attr_gamma['a']).to(self.device) * d_minus

            p_comp = F.softmax(d_minus_comp - d_plus_comp, dim=1)
            p_attr = F.softmax(d_minus_attr - d_plus_attr, dim=1)

            return p_comp, p_attr
