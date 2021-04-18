import numpy as np
import os, logging, torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(prob, labels, depth, target=1, weight=None, sample_weight=None, neg_loss=0, focal_loss=0):
    """cross entropy with GT label id (int)"""
    onehot_label = one_hot(labels, depth=depth, axis=1)  # (bz, depth)
    return cross_entropy_with_labelvec(prob, onehot_label, target=target,
                                       weight=weight, sample_weight=sample_weight, neg_loss=neg_loss)


def cross_entropy_with_labelvec(prob, label_vecs, target=1, weight=None, sample_weight=None, neg_loss=0, mask=None,
                                focal_loss=0):
    """cross entropy with GT onehot vector"""
    assert target in [0, 1]
    epsilon = 1e-8
    gamma = focal_loss
    alpha = neg_loss

    zeros = torch.zeros_like(label_vecs, dtype=prob.dtype)  # (bz, depth)
    ones = torch.ones_like(label_vecs, dtype=prob.dtype)  # (bz, depth)

    if target == 0:
        prob = 1 - prob

    pos_position = label_vecs > torch.zeros_like(label_vecs, dtype=label_vecs.dtype)
    pos_prob = torch.where(pos_position, prob, ones)  # pos -> p, neg -> 1 (no loss)
    neg_prob = torch.where(pos_position, zeros, prob)  # pos -> 0(no loss), neg ->p

    pos_xent = - torch.log(torch.clamp(pos_prob, epsilon, 1.0))
    if gamma is not None:
        pos_xent = ((1 - pos_prob) ** gamma) * pos_xent

    if alpha is None or alpha == 0:
        neg_xent = 0
    else:
        neg_xent = - alpha * torch.log(torch.clamp(1.0 - neg_prob, epsilon, 1.0))

        if gamma is not None:
            neg_xent = (neg_prob ** gamma) * neg_xent

    xent = pos_xent + neg_xent  # (bz, depth)

    if mask is not None:
        xent = xent * mask

    if sample_weight is not None:
        xent = xent * torch.unsqueeze(sample_weight, dim=1)
    xent = torch.mean(xent, dim=0)  # (depth,)

    if weight is not None:
        xent = xent * weight

    return torch.sum(xent)


def one_hot(labels, depth, device=torch.device('cpu'), axis=1):
    assert axis == 1
    batch_size = labels.shape[0]
    labels_onehot = torch.FloatTensor(batch_size, depth).to(device)
    labels_onehot.zero_()
    labels_onehot.scatter_(axis, labels.reshape(batch_size, 1), 1)
    return labels_onehot


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
    dist = self.distance_metric(anchor, positive) - self.distance_metric(anchor, negative) + self.args.triplet_margin
    return torch.max(dist, 0)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, args, is_training, name=None, hidden_layers=[]):
        super(MLP, self).__init__()
        self.name = name

        activation_list = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
        }
        if hasattr(nn, "LeakyReLU"):
            activation_list['leaky_relu'] = nn.LeakyReLU()
        if hasattr(nn, "ReLU6"):
            activation_list['relu6'] = nn.ReLU6()

        self.activation_fn = activation_list[args.activation]
        # self.normalizer_fn = F.batch_norm if args.batchnorm else None
        # self.normalizer_params = {
        #     'is_trainning': is_training,
        #     'decay': 0.95,
        #     'fused': False
        # }

        self.dropout = args.dropout

        self.layers = nn.ModuleList()
        pre_dim = input_dim
        for cur_dim in hidden_layers:
            linear = nn.Linear(pre_dim, cur_dim)
            self.layers.append(linear)
            nn.init.kaiming_uniform_(linear.weight)
            if args.batchnorm:
                self.layers.append(nn.BatchNorm1d(cur_dim, momentum=0.05))
            self.layers.append(self.activation_fn)
            if args.dropout is not None:
                self.layers.append(nn.Dropout(cur_dim))

            pre_dim = cur_dim

        self.layers.append(nn.Linear(pre_dim, output_dim))

    def forward(self, input_feat):
        for layer in self.layers:
            input_feat = layer(input_feat)

        return input_feat

    def string(self):
        return self.name


class BaseNetwork(nn.Module):
    def __init__(self, dataloader, args, feat_dim=None, device=torch.device('cpu')):
        super(BaseNetwork, self).__init__()

        self.dataloader = dataloader
        self.num_attr = len(dataloader.dataset.attrs)
        self.num_obj = len(dataloader.dataset.objs)
        if feat_dim is not None:
            self.feat_dim = feat_dim
        else:
            self.feat_dim = dataloader.dataset.feat_dim

        self.device = device
        self.args = args
        self.dropout = args.dropout

        if self.args.loss_class_weight:
            from utils.aux_data import load_loss_weight

            self.num_pair = len(dataloader.dataset.pairs)
            attr_weight, obj_weight, pair_weight = load_loss_weight(self.args.data)
            self.attr_weight = torch.from_numpy(np.array(attr_weight, dtype=np.float32)).to(self.device)
            self.obj_weight = torch.from_numpy(np.array(obj_weight, dtype=np.float32)).to(self.device)
            self.pair_weight = torch.from_numpy(np.array(pair_weight, dtype=np.float32)).to(self.device)

        else:
            self.attr_weight = None
            self.obj_weight = None
            self.pair_weight = None

    def logger(self, suffix):
        return self.root_logger.getChild(suffix)

    # def basic_argscope(self, is_training):
    #     activation_list = {
    #         'relu': nn.ReLU,
    #         'elu': nn.ELU,
    #     }
    #     if hasattr(nn, "LeakyReLU"):
    #         activation_list['leaky_relu'] = nn.LeakyReLU
    #     if hasattr(nn, "ReLU6"):
    #         activation_list['relu6'] = nn.ReLU6

    #     """
    #     if self.args.initializer is None:
    #         initializer = tf.contrib.layers.xavier_initializer()
    #     else:
    #         initializer = tf.random_normal_initializer(0, self.args.initializer)
    #     """

    #     return None

    #     """        
    #     slim.arg_scope(
    #         [slim.fully_connected],
    #         activation_fn = activation_list[self.args.activation],
    #         normalizer_fn = slim.batch_norm if self.args.batchnorm else None,
    #         normalizer_params={
    #             'is_training': is_training, 
    #             'decay': 0.95, 
    #             'fused':False
    #         },
    #     )
    #     """

    # def MLP(self, input_feat, output_dim, is_training, name, hidden_layers=[]):
    #     """multi-layer perceptron, 1 layers as default"""

    #     with self.basic_argscope(is_training):
    #         with tf.variable_scope(name) as scope:
    #             for i, size in enumerate(hidden_layers):
    #                 input_feat = slim.fully_connected(input_feat, size,
    #                     trainable=is_training, reuse=tf.AUTO_REUSE, scope="fc_%d"%i)

    #                 if self.dropout is not None:
    #                     input_feat = slim.dropout(input_feat, keep_prob=self.dropout, is_training=is_training, scope='dropout_%d'%i)

    #             output_feat = slim.fully_connected(input_feat, output_dim,
    #                 trainable=is_training, activation_fn=None, reuse=tf.AUTO_REUSE, scope="fc_out")

    #     return output_feat

    # def test_step(self, score_op):
    #     dset = self.dataloader.dataset
    #     test_att = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
    #     test_obj = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])
    #
    #     feed_dict = {
    #         self.pos_image_feat: blobs[4],
    #         self.test_attr_id: test_att,
    #         self.test_obj_id: test_obj,
    #     }
    #     if self.args.obj_pred is not None:
    #         feed_dict[self.pos_obj_prediction] = blobs[-1]
    #
    #     score = sess.run(score_op, feed_dict=feed_dict)
    #
    #     for key in score_op.keys():
    #         score[key][0] = {
    #             (a, o): torch.from_numpy(score[key][0][:, i])
    #             for i, (a, o) in enumerate(zip(test_att, test_obj))
    #         }
    #
    #     return score
