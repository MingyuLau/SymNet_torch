from .. import config as cfg
from . import CZSL_dataset, GCZSL_dataset
from torch.utils.data import DataLoader
import numpy as np


def get_dataloader(dataset_name, phase, feature_file="features.t7", batchsize=1, num_workers=1, shuffle=None, **kwargs):
    
    if dataset_name in ["MITg", "UTg"]:
        # dataset_name = dataset_name[:-1]
        dataset =  GCZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = cfg.GCZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            **kwargs)
    else:
        dataset =  CZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = cfg.CZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            **kwargs)
    

    if shuffle is None:
        shuffle = (phase=='train')
    
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)


    

