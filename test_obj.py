import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import numpy as np
import tqdm
import os
import os.path as osp
import logging
import importlib
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter


from utils import config as cfg
from utils import dataset, utils
from utils.evaluator import CZSL_Evaluator

from run_symnet import make_parser


def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    parser = make_parser()
    parser.add_argument("--test_set", type=str, default='test',
        choices=['test','val'])
    args = parser.parse_args()
    utils.display_args(args, logger)


    # logging and pretrained weight dirs
    log_dir = osp.join(cfg.LOG_ROOT_DIR, args.name)


    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, args.test_set, 
        batchsize=args.test_bz)
    

    logger.info("Loading network")
    network_module = importlib.import_module('models.'+args.network)
    model = network_module.Model(test_dataloader.dataset, args).cuda()
    print(model)

    


    # initialization (model weight, optimizer, lr_scheduler, clear logs)

    
    if args.trained_weight is None:
        # load weight
        args.trained_weight = utils.CheckpointPath.compose(log_dir, args.epoch)
        checkpoint = torch.load(args.trained_weight)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Checkpoint <= "+args.trained_weight)
    else:
        raise ValueError("Do not specify checkpoint path")


    # evaluator
    evaluator = CZSL_Evaluator(test_dataloader.dataset, model)


    
    # save obj predictions
    pklname = "%s_%s_ep%d.pt"%(args.name, args.test_set, args.epoch)
    pklpath = osp.join(cfg.DATA_ROOT_DIR, "obj_scores", pklname)
    assert args.force or not os.path.exists(pklpath), pklpath
    logger.info("obj prediction => "+pklpath)

    with torch.no_grad():
        test_epoch(model, evaluator, test_dataloader, pklpath)

    logger.info('Finished.')




def test_epoch(model, evaluator, dataloader, pklpath):
    accuracies_obj = []
    predictions_obj = []

    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k:v.cuda() for k,v in batch.items()}

        pred_attr, pred_obj = model(batch, require_loss=False)

        attr_truth, obj_truth = batch["pos_attr_id"], batch["pos_obj_id"]

        _, o_match = evaluator.evaluate_only_attr_obj(
            pred_attr, attr_truth, pred_obj, obj_truth)

        accuracies_obj.append(o_match)
        predictions_obj.append(pred_obj)
    
    predictions_obj = torch.cat(predictions_obj).cpu()


    print("Saving %s tensor to %s"%(str(predictions_obj.size()), pklpath))
    torch.save(predictions_obj, pklpath)


    real_obj_acc = torch.mean(torch.cat(accuracies_obj)).item()
    print("Obj acc = %.4f"%(real_obj_acc*100.))




if __name__=="__main__":
    main()
