import importlib

def load_loss_weight(dataset_name: str) -> (list, list):
    """Loss weight to balance the categories
    weight = -log(frequency)"""
    
    try:
        Weight = importlib.import_module('utils.aux_data.%s_weight'%dataset_name)
        return Weight.attr_weight, Weight.obj_weight
        
    except ImportError:
        raise NotImplementedError("Loss weight for %s is not implemented yet"%dataset_name)


def load_wordvec_dict(dataset_name: str, vec_type: str) -> (dict, dict):
    dsname_mapping = {
        "MITg": "MIT",
        "UTg": "UT",
    }
    if dataset_name in dsname_mapping:
        dataset_name = dsname_mapping[dataset_name]
        
    try:
        Wordvec = importlib.import_module('utils.aux_data.%s_%s'%(vec_type, dataset_name))
        return Wordvec.attrs_dict, Wordvec.objs_dict

    except ImportError:
        raise NotImplementedError("%s vector for %s is not ready yet"%(vec_type, dataset_name))
