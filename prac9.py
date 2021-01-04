"""
test for model load and save
"""

import torch
from data.load_dataset import CustomerDataLoader
from lib.models.image_transfer import resize_image
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err
from lib.models.metric_depth_model import *
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue
import math
import traceback
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions


path="/home/colin/papercode/VNL_Monocular_Depth_Prediction-master/tools/outputs/Jan04-21-55-14_colin-Alienware-Aurora-R7/ckpt/epoch0_step10.pth"
checkpoint = torch.load(path)

print(checkpoint['model_state_dict'].keys())


