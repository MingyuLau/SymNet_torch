import time
import cv2
import os
import argparse
import os.path as osp
from PIL import Image
import numpy as np
import torch

from . import config as cfg
from . import aux_data

# Save the training script and all the arguments to a file so that you 
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(args):
    shutil.copy('train.py', args.cv_dir)
    shutil.copy('models/models.py', args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))



################################################################################
#                               tools for solvers                              #
################################################################################

def display_args(args, logger, verbose=False):
    """print some essential arguments"""
    if verbose:
        ignore = []
        for k,v in args.__dict__.items():
            if not callable(v) and not k.startswith('__') and k not in ignore:
                logger.info("{:30s}{}".format(k,v))
    else:
        logger.info('Name:       %s'%args.name)
        logger.info('Network:    %s'%args.network)
        logger.info('Data:       %s'%args.data)
        logger.info('FC layers:  At {fc_att}, Cm {fc_compress}, Cls {fc_cls}'.format(
            **args.__dict__))



def duplication_check(args):
    if args.force:
        return
    elif args.trained_weight is None or args.trained_weight.split('/')[0] != args.name:
        assert not osp.exists(osp.join(cfg.WEIGHT_ROOT_DIR, args.name)), \
            "weight dir with same name exists (%s)"%(args.name)
        assert not osp.exists(osp.join(cfg.LOG_ROOT_DIR, args.name)), \
            "log dir with same name exists (%s)"%(args.name)
        

def formated_czsl_result(report):
    fstr = '[{name}/{epoch}] rA:{real_attr_acc:.4f}|rO:{real_obj_acc:.4f}|Cl/T1:{top1_acc:.4f}|T2:{top2_acc:.4f}|T3:{top3_acc:.4f}'

    return fstr.format(**report)


################################################################################
#                                glove embedder                                #
################################################################################

class Embedder:
    """word embedder (for various vector type)
    __init__(self)
    """

    def __init__(self, vec_type, vocab, data):
        self.vec_type = vec_type
        self.device = device

        if vec_type != 'onehot':
            self.embeds = self.load_word_embeddings(vec_type, vocab, data)
            self.emb_dim = self.embeds.shape[1]
        else:
            self.emb_dim = len(vocab)
    
    def get_embedding(self, i):
        """actually implements __getitem__() function"""
        if self.vec_type == 'onehot':
            return one_hot(i, depth=self.emb_dim, axis=1).to(self.device)
        else:
            i_onehot = one_hot(i, depth=self.embeds.shape[0], axis=1)
            return torch.matmul(i_onehot.to(self.device),
                                torch.from_numpy(self.embeds).to(self.device))

    def load_word_embeddings(self, vec_type, vocab, data):
        tmp = aux_data.load_wordvec_dict(data, vec_type)
        if type(tmp) == tuple:
            attr_dict, obj_dict = tmp
            attr_dict.update(obj_dict)
            embeds = attr_dict
        else:
            embeds = tmp

        embeds_list = []
        for k in vocab:
            if k in embeds:
                embeds_list.append(embeds[k])
            else:
                raise NotImplementedError('some vocabs are not in dictionary: %s'%k)

        embeds = np.array(embeds_list, dtype=np.float32)

        print ('Embeddings shape = %s'%str(embeds.shape))
        return embeds


################################################################################
#                                network utils                                 #
################################################################################

# def one_hot(tensor, depth, axis):
#     assert len(tensor.shape) == 1
#     assert axis == 1
#     size = tensor.shape[0]
#     ans = torch.zeros(size, depth)
#     ans[torch.arange(size), tensor] = 1
#     return ans


# def repeat_tensor(tensor, axis, multiple):
#     raise NotImplementedError("you can use torch.repeat()")
#     """e.g. (1,2,3)x3 = (1,1,1,2,2,2,3,3,3)"""
    
#     result_shape = list(tensor.shape)
#     for i, v in enumerate(result_shape):
#         if v is None:
#             result_shape[i] = tensor.shape[i]
#     result_shape[axis] *= multiple

#     tensor = torch.unsqueeze(tensor, axis+1)
#     mul = [1] * len(tensor.shape)
#     mul[axis+1] = multiple
#     tensor = tensor.repeat(mul)
#     tensor = torch.reshape(tensor, result_shape)

#     return tensor


# def tile_tensor(tensor, axis, multiple):
#     """e.g. (1,2,3)x3 = (1,2,3,1,2,3,1,2,3)"""
#     raise NotImplementedError("you can use torch.tile()")
#     mul = [1] * len(tensor.shape)
#     mul[axis] = multiple

#     return tensor.repeat(mul)


def activation_func(name):
    if name == "none":
        return (lambda x:x)
    elif name == "sigmoid":
        return tf.sigmoid
    elif name == "relu":
        return tf.nn.relu
    else:
        raise NotImplementedError("activation function %s not implemented"%name)

