import logging, os

logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)


SUMMARY_INTERVAL        = 'auto' # int(of iter) or 'auto'
IMAGE_SUMMARY_INTERVAL  = 'auto' # int(of iter) or 'auto'


ROOT_DIR = "."   # change this to the project folder
LOG_ROOT_DIR          = ROOT_DIR+"/logs/"
DATA_ROOT_DIR         = ROOT_DIR+"/data"


CZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-original',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-original',
}
GCZSL_DS_ROOT = {
    'MITg': DATA_ROOT_DIR+'/mit-states-natural',
    'UTg':  DATA_ROOT_DIR+'/ut-zap50k-natural',
}


GRADIENT_CLIPPING = 5


os.makedirs(LOG_ROOT_DIR, exist_ok=True)