import os
from pathlib import Path

ROOT = f"{str(Path(__file__).parent)}/../../../download"
#ROOT = f"{str(Path(__file__).parent)}/download"
KMEANS_WEIGHT_ROOT = f"{ROOT}/clustering_weights"
# KMEANS_WEIGHT_ROOT = "./download/clustering_weights"
DATASET_DIR = f"{ROOT}/datasets"
FID_WEIGHT_DIR = f"{ROOT}/fid_weights/FIDNetV3"
JOB_DIR = f"{ROOT}/pretrained_weights"


# CANVAS
SIZE = (360, 240)
CANVAS_WIDTH = 1200
CANVAS_LENGTH = 1200


# DATA
HOME_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR  = os.path.abspath(os.path.join(HOME_DIR, 'input'))
UTILS_DIR = os.path.abspath(os.path.join(HOME_DIR, 'utils'))
FONTS_DIR = os.path.abspath(os.path.join(UTILS_DIR, 'fonts'))
IMAGE_FILE_TYPES = ('.png', '.jpg')


# OUTPUT
OUTPUT_DIR = os.path.abspath(os.path.join(HOME_DIR, 'output'))


# for model
MODEL_LABELS = {
    'layoutdm_publaynet':   ['text', 'title', 'list', 'table', 'figure'],
    'layoutdm_rico':        ['Text', 'Image', 'Icon', 'Text Button', 'List Item','Input', 'Background Image', 'Card', 'Web View', 'Radio Button',
        'Drawer', 'Checkbox', 'Advertisement', 'Modal', 'Pager Indicator', 'Slider',
        'On/Off Switch', 'Button Bar', 'Toolbar', 'Number Stepper', 'Multi-Tab',
        'Date Picker', 'Map View', 'Video', 'Bottom Navigation']
    }
LABEL_MAP = {
    'backgrounds':  {'layoutdm_publaynet': 4, 'layoutdm_rico': 6},
    'images':       {'layoutdm_publaynet': 4, 'layoutdm_rico': 1},
    'headers':      {'layoutdm_publaynet': 0, 'layoutdm_rico': 0},
    'texts':        {'layoutdm_publaynet': 0, 'layoutdm_rico': 0}
    }
