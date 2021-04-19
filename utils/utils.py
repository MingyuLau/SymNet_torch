import time
import cv2
import parse
import re
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



def duplication_check(args, log_dir):
    if args.force:
        return
    elif args.trained_weight is None or not utils.CheckpointPath.in_dir(args.trained_weight, log_dir)
        assert not osp.exists(log_dir, "log dir with same name exists (%s)"%log_dir)
        

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
        raise NotImplementedError("Activation function {} is not implemented".format(name))


def get_optimizer(optim_type, lr, params):
    if optim_type == 'sgd':
        logger = logging.getLogger('utils.get_optimizer')
        logger.info('Using {} optimizer'.format(optim_type))

    if optim_type == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif optim_type == 'momentum':
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif optim_type == 'adam':
        return torch.optim.Adam(params, lr=lr)
    elif optim_type == 'adamw':
        return torch.AdamW(params, eps=5e-5, lr=lr)
    elif optim_type == 'rmsprop':
        return torch.RMSprop(params, lr=lr)
    else:
        raise NotImplementedError("{} optimizer is not implemented".format(optim_type))


def set_scheduler(args, optimizer, train_dataloader):
    """ return a learning rate scheduler """
    decay_stepsize = len(train_dataloader) * args.lr_decay_step

    if args.lr_decay_type == 'no':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 1.0))
    elif args.lr_decay_type == 'exp':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=args.lr_decay_rate)
    elif args.lr_decay_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=decay_stepsize, T_mult=2, eta_min=0)
        # TODO: this is NOT equivalent to the tf version

    # elif opt.lr_policy == 'linear':
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # elif opt.lr_policy == 'step':
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # elif opt.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # elif opt.lr_policy == 'cosine':
    #     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=opt.niter, eta_min=0)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_decay_type)
    return scheduler

    
def clear_folder(dirname):
    """clear weight and log dir"""
    logger = self.logger('clear_folder')

    for f in os.listdir(dirname):
        logger.warning('Deleted log file ' + f)
        os.remove(os.path.join(dirname, f))



class CheckpointPath(object):
    TEMPLATE = "{:s}/checkpoint_ep{:d}.pt"
    EPOCH_PATTERN = "checkpoint_ep([0-9]*).pt"

    @classmethod
    def compose(log_dir: str, epoch: int) -> str:
        return TEMPLATE.format(log_dir, epoch)

    @classmethod
    def decompose(ckpt_path: str) -> (str, int):
        log_dir = osp.dirname(ckpt_path)
        epoch = re.match(EPOCH_PATTERN, osp.basename(ckpt_path)).group(0)
        return log_dir, int(epoch)
    
    @classmethod
    def in_dir(ckpt_path: str, log_dir: str) -> bool:
        ckpt_log_dir, _ = decompose(ckpt_path)
        return osp.samefile(ckpt_log_dir, log_dir)


def snapshot(model, log_dir, epoch):
    """save checkpoint"""
    ckpt_path = CheckpointPath.compose(log_dir, epoch)
    
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer,
    }, ckpt_path)

    logging.getLogger('utils.snapshot').info('Wrote snapshot to: {}'.format(ckpt_path))
