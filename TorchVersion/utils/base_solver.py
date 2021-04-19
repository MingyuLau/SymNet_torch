from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import torch
from torch.optim import lr_scheduler

from utils import config as cfg


class BaseSolver(object):
    root_logger = logging.getLogger('solver')

    def __init__(self):
        self.train_dataloader = None
        self.args = None
        self.trained_weight = None
        self.weight_dir = None
        self.log_dir = None

    def logger(self, suffix):
        return self.root_logger.getChild(suffix)


    def snapshot(self, model, iter, filenames=None):
        """save checkpoint"""
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        if filenames is None:
            filename = 'snapshot_epoch_{}.ckpt'.format(iter)
        else:
            filename = filenames
        pth = os.path.join(self.weight_dir, filename)
        torch.save(model.state_dict(), pth)

        self.logger('snapshot').info('Wrote snapshot to: {}'.format(filename))



    def set_scheduler(self, optimizer):
        """ return a learning rate scheduler """
        decay_stepsize = len(self.train_dataloader) * self.args.lr_decay_step

        if self.args.lr_decay_type == 'no':
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 1.0))
        elif self.args.lr_decay_type == 'exp':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=self.args.lr_decay_rate)
        elif self.args.lr_decay_type == 'cos':
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
            return NotImplementedError('learning rate policy [%s] is not implemented', self.args.lr_decay_type)
        return scheduler


