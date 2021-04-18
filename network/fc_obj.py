from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, logging, torch
from collections import OrderedDict

from network.base_network import *
from utils.utils import Embedder
from utils import config as cfg


class NetWork(BaseNetwork):
    def __init__(self, dataloader, args, feat_dim, device=torch.device('cpu')):
        super().__init__(dataloader, args, feat_dim, device)

        self.feat_dim = feat_dim
        self.obj_cls_mlp_layer = MLP(feat_dim, self.num_obj, self.args, is_training=True, name='obj_cls',
                                     hidden_layers=self.args.fc_cls)

    def forward(self, blobs):
        pos_obj_id = torch.from_numpy(blobs[2]).to(self.device)
        pos_image_feat = torch.from_numpy(blobs[9]).to(self.device)  # why different from symnet??

        ########################## classifier losses ##########################
        score_pos_O = self.obj_cls_mlp_layer(pos_image_feat)
        loss = F.cross_entropy(score_pos_O, pos_obj_id, weight=self.obj_weight)
        # TODO: focal loss not implemented

        # prob_pos_O = F.softmax(score_pos_O, 1)
        # loss = cross_entropy(prob_pos_O, pos_obj_id,
        #                      depth=self.num_obj, weight=self.obj_weight, focal_loss=self.args.focal_loss)

        train_summary_op = OrderedDict()
        train_summary_op['loss'] = loss

        return loss, train_summary_op

    def test_step(self, blobs):
        dset = self.dataloader.dataset
        test_att = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
        test_obj = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])
        pos_image_feat = torch.from_numpy(blobs[4]).to(self.device)
        if self.args.obj_pred is not None:
            pos_obj_prediction = torch.from_numpy(blobs[-1]).to(self.device)

        score_pos_O = self.obj_cls_mlp_layer(pos_image_feat)
        prob_pos_O = F.softmax(score_pos_O, 1)

        batchsize = pos_image_feat.shape[0]
        prob_pos_A = torch.zeros([batchsize, self.num_attr],
                                 dtype=pos_image_feat.dtype).to(self.device)
        score_original = torch.zeros([batchsize, self.num_pair],
                                     dtype=pos_image_feat.dtype).to(self.device)

        score = OrderedDict([
            ("score_fc", [score_original, prob_pos_A, prob_pos_O]),
        ])

        for key in score.keys():
            score[key][0] = {
                (a, o): score[key][0][:, i]
                for i, (a, o) in enumerate(zip(test_att, test_obj))
            }

        return score


"""
class Network(BaseNetwork):
    root_logger = logging.getLogger("network %s"%__file__)

    def __init__(self, dataloader, args, feat_dim=None):
        super(Network, self).__init__(dataloader, args, feat_dim)

        self.pos_obj_id   = tf.placeholder(tf.int32, shape=[None])
        self.pos_image_feat = tf.placeholder(tf.float32, shape=[None, self.feat_dim])
        self.lr = tf.placeholder(tf.float32)
    



    def build_network(self, test_only=False):
        logger = self.logger('create_train_arch')


        ########################## classifier losses ##########################
        score_pos_O = self.MLP(self.pos_image_feat, self.num_obj, 
            is_training=True, name='obj_cls', 
            hidden_layers=self.args.fc_cls)
        prob_pos_O = tf.nn.softmax(score_pos_O, 1)
        
        loss = self.cross_entropy(prob_pos_O, self.pos_obj_id, 
            depth=self.num_obj, weight=self.obj_weight)


        ################################ test #################################
        
        score_pos_O = self.MLP(self.pos_image_feat, self.num_obj,
            is_training=False, name='obj_cls', 
            hidden_layers=self.args.fc_cls)
        prob_pos_O = tf.nn.softmax(score_pos_O, 1)

        batchsize = tf.shape(self.pos_image_feat)[0]
        prob_pos_A = tf.zeros([batchsize, self.num_attr],
            dtype=self.pos_image_feat.dtype)
        score_original = tf.zeros([batchsize, self.num_pair],
            dtype=self.pos_image_feat.dtype)

        score_res = OrderedDict([
            ("score_fc", [score_original, prob_pos_A, prob_pos_O]),
        ])
        

        # summary
        
        with tf.device("/cpu:0"):
            tf.summary.scalar('loss_total', loss)
            tf.summary.scalar('lr', self.lr)
            
        
        train_summary_op = tf.summary.merge_all()


        if test_only:
            return prob_pos_O
        else:
            return loss, score_res, train_summary_op



    def train_step(self, sess, blobs, lr, train_op, train_summary_op):
        summary, _ = sess.run(
            [train_summary_op, train_op],
            feed_dict={
                self.pos_obj_id      : blobs[2],
                self.pos_image_feat  : blobs[9],
                self.lr: lr,
            })

        return summary

"""
