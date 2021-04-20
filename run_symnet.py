import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import numpy as np
import tqdm
import os
import os.path as osp
import itertools
import glob


import logging
import importlib
import re
import copy
import random
import argparse
import pickle
from pprint import pprint
from datetime import datetime
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


from utils import config as cfg
from utils import dataset, utils
from utils.evaluator import CZSL_Evaluator



def make_parser():
    parser = argparse.ArgumentParser()

    # basic training settings
    parser.add_argument("--name", type=str, required=True, 
        help="Experiment name")
    parser.add_argument("--data", type=str, required=True,
        choices=['MIT','UT','MITg','UTg'],
        help="Dataset name")
    parser.add_argument("--network", type=str, default='symnet', 
        help="Network name (the file name in `network` folder, but without suffix `.py`)")


    parser.add_argument("--epoch", type=int, default=500,
        help="Maximum epoch during training phase, or epoch to be tested during testing")
    parser.add_argument("--bz", type=int, default=512,
        help="Train batch size")
    parser.add_argument("--test_bz", type=int, default=1024,
        help="Test batch size")

    parser.add_argument("--trained_weight", type=str, default=None,
        help="Restore from a certain trained weight (relative path to './weights')")
    parser.add_argument("--weight_type", type=str, default="continue", 
        help="Type of the trained weight: 'continue'-previous checkpoint(default), ?-pretrained classifier, ?-pretrained transformer")

    parser.add_argument("--test_freq", type=int, default=1,
        help="Frequency of testing (#epoch). Set to 0 to skip test phase")
    parser.add_argument("--snapshot_freq", type=int, default=10,
        help="Frequency of saving snapshots (#epoch)")

    parser.add_argument("--force", default=False, action='store_true',
        help="WARINING: clear experiment with duplicated name")
    
    
    # model settings

    parser.add_argument("--rmd_metric", type=str, default='softmax',
        help="Similarity metric in RMD classification")
    parser.add_argument("--distance_metric", type=str, default='L2', 
        help="Distance form")

    parser.add_argument("--obj_pred", type=str, default=None, 
        help="Object prediction from pretrained model")

    parser.add_argument("--wordvec", type=str, default='glove',
        help="Pre-extracted word vector type")
    


    # important hyper-parameters
    
    parser.add_argument("--rep_dim", type=int, default=300,
        help="Dimentionality of attribute/object representation")
    
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate")

    parser.add_argument("--dropout", type=float, 
        help="Keep probability of dropout")
    parser.add_argument("--batchnorm", default=False, action='store_true',
        help="Use batch normalization")
    parser.add_argument("--loss_class_weight", default=True, action='store_true', 
        help="Add weight between classes (default=true)")
        
    parser.add_argument("--fc_att", type=int, default=[512], nargs='*',
        help="#fc layers after word vector")
    parser.add_argument("--fc_compress", type=int, default=[768], nargs='*',
        help="#fc layers after hidden layer")
    parser.add_argument("--fc_cls", type=int, default=[512], nargs='*',
        help="#fc layers in classifiers")
    
    
    
    parser.add_argument("--lambda_cls_attr", type=float, default=0)
    parser.add_argument("--lambda_cls_obj", type=float, default=0)

    parser.add_argument("--lambda_trip", type=float, default=0)
    parser.add_argument("--triplet_margin", type=float, default=0.5,
        help="Triplet loss margin")

    parser.add_argument("--lambda_sym", type=float, default=0)

    parser.add_argument("--lambda_axiom", type=float, default=0)
    parser.add_argument("--remove_inv", default=False, action="store_true")
    parser.add_argument("--remove_com", default=False, action="store_true")
    parser.add_argument("--remove_clo", default=False, action="store_true")

    parser.add_argument("--no_attention", default=False, action="store_true")
    


    # not so important
    parser.add_argument("--activation", type=str, default='relu',
        help="Activation function (relu, elu, leaky_relu, relu6)")
    parser.add_argument("--initializer", type=float, default=None,
        help="Weight initializer for fc (default=xavier init, set a float number to use Gaussian init)")
    parser.add_argument("--optimizer", type=str, default='sgd', 
        choices=['sgd', 'adam', 'adamw', 'momentum', 'rmsprop'],
        help="Type of optimizer (sgd, adam, momentum)")

    parser.add_argument("--lr_decay_type", type=str, default='no',
        help="Type of Learning rate decay: no/exp/cos")
    parser.add_argument("--lr_decay_step", type=int, default=100,
        help="The first learning rate decay step (only for cos/exp)")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9,
        help="Decay rate of cos/exp")

    parser.add_argument("--focal_loss", type=float,
        help="`gamma` in focal loss. default=0 (=CE)")
    
    parser.add_argument("--clip_grad", default=False, action='store_true',
        help="Use gradient clipping")

    
    return parser





################################################################################


def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    parser = make_parser()
    args = parser.parse_args()
    utils.display_args(args, logger)


    # logging and pretrained weight dirs
    log_dir = osp.join(cfg.LOG_ROOT_DIR, args.name)
    utils.duplication_check(args, log_dir)
    logger.info("Training ckpt and log  => "+log_dir)
    os.makedirs(log_dir, exist_ok=True)


    logger.info("Loading dataset")
    train_dataloader = dataset.get_dataloader(args.data, 'train', 
        batchsize=args.bz)
    test_dataloader = dataset.get_dataloader(args.data, 'test', 
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    

    logger.info("Loading network")
    network_module = importlib.import_module('models.'+args.network)
    model = network_module.Model(train_dataloader.dataset, args).cuda()
    print(model)

    


    # initialization (model weight, optimizer, lr_scheduler, clear logs)

    if args.trained_weight is None:
        # from scratch
        init_epoch = 0
        utils.clear_folder(log_dir)
        
        # optimizer
        # todo: different lr on different params (refer to AttrOp code)
        params = model.parameters()
        optimizer = utils.get_optimizer(args.optimizer, args.lr, params)
        # TODO: gradient clipping

        # lr scheduler
        # TODO!
    else:
        # load weight
        checkpoint = torch.load(args.trained_weight)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']

        logger.info("Checkpoint <= "+args.trained_weight)
        
        if args.weight_type == "continue":
            init_epoch = checkpoint["epoch"]
        else:
            utils.clear_folder(log_dir)


    # evaluator
    evaluator = CZSL_Evaluator(test_dataloader.dataset, model)
    if args.network == 'fc_obj':
        main_score_key = 'real_obj_acc'
    else:
        main_score_key = 'top1_acc'
    best_report = None

    # logger
    writer = SummaryWriter(log_dir)


    # trainval
    logger.info('Start training')
    for epoch in range(init_epoch+1, args.epoch+1):
        train_epoch(model, optimizer, train_dataloader, writer, epoch, args.epoch)

        if args.test_freq>0 and epoch%args.test_freq == 0:
            with torch.no_grad():
                current_report = test_epoch(model, evaluator, test_dataloader, writer, epoch)

            if best_report is None or current_report[main_score_key]>best_report[main_score_key]:
                best_report = current_report
            
            # print test results
            print("Current: " + utils.formated_czsl_result(current_report))
            print("Best: " + utils.formated_czsl_result(best_report))

        
        if args.snapshot_freq>0 and epoch%args.snapshot_freq == 0:
            utils.snapshot(model, log_dir, optimizer, epoch)


    writer.close()
    logger.info('Finished.')



##########################################################################

def train_epoch(model, optimizer, dataloader, writer, epoch, max_epoch):
    model.train()

    total_loss = defaultdict(list)

    for batch_ind, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix='Train %d/%d'%(epoch, max_epoch)):
        batch = {k:v.cuda() for k,v in batch.items()}
        
        _, _, losses = model(batch)

        optimizer.zero_grad()
        losses["loss_total"].backward()
        optimizer.step()

        for key, value in losses.items():
            total_loss[key].append(value.item())
    

    for key, value in total_loss.items():
        writer.add_scalar(key, np.mean(value), epoch)
        # print(key, np.mean(value))



def test_epoch(model, evaluator, dataloader, writer, epoch):
    accuracies_pair = []
    accuracies_attr = []
    accuracies_obj = []

    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix='Test %d'%epoch):
        batch = {k:v.cuda() for k,v in batch.items()}

        pred_attr, pred_obj = model(batch, require_loss=False)

        attr_truth, obj_truth = batch["pos_attr_id"], batch["pos_obj_id"]
        pred_pair = utils.generate_pair_result(pred_attr, pred_obj, dataloader.dataset)
        pair_results = evaluator.score_model(pred_pair, obj_truth)
        match_stats = evaluator.evaluate_predictions(
            pair_results, attr_truth, obj_truth)
        accuracies_pair.append(match_stats)
        # 0/1 sequence of t/f

        a_match, o_match = evaluator.evaluate_only_attr_obj(
            pred_attr, attr_truth, pred_obj, obj_truth)

        accuracies_attr.append(a_match)
        accuracies_obj.append(o_match)


    accuracies = accuracies_pair
    accuracies = zip(*accuracies)
    accuracies = map(torch.mean, map(torch.cat, accuracies))
    attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, _ = map(lambda x:x.item(), accuracies)

    real_attr_acc = torch.mean(torch.cat(accuracies_attr)).item()
    real_obj_acc = torch.mean(torch.cat(accuracies_obj)).item()

    report_dict = {
        'real_attr_acc':real_attr_acc,
        'real_obj_acc': real_obj_acc,
        'top1_acc':     closed_1_acc,
        'top2_acc':     closed_2_acc,
        'top3_acc':     closed_3_acc,
        'epoch':        epoch,
    }

    # save to tensorboard
    if writer:
        for key, value in report_dict.items():
            if key not in ['name', 'epoch']:
                writer.add_scalar("score/"+key, value, epoch)

    return report_dict


if __name__=="__main__":
    main()
