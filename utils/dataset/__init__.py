from .. import config as cfg
from . import CZSL_dataset, GCZSL_dataset
from torch.utils.data import DataLoader
import numpy as np


def get_dataloader(dataset_name: str, phase:str, 
                    feature_file: str="features.t7", 
                    batchsize: int=1, num_workers: int=1, 
                    shuffle: bool=None, **kwargs) -> DataLoader:

    
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


    

