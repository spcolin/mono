"""
test for the load of train dataset
"""


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



if __name__=='__main__':
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()
    # train_opt.print_options(train_args)

    # # Validation args
    # val_opt = ValOptions()
    # val_args = val_opt.parse()
    # val_args.batchsize = 1
    # val_args.thread = 0
    # # val_opt.print_options(val_args)

    train_dataloader = CustomerDataLoader(train_args)
    train_datasize = len(train_dataloader)
    gpu_num = torch.cuda.device_count()
    merge_cfg_from_file(train_args)

    # val_dataloader = CustomerDataLoader(val_args)
    # val_datasize = len(val_dataloader)

    # data=None
    # for i, data in enumerate(train_dataloader):
    #     print(data.keys())
    #     break



    # # Print configs
    # print_configs(cfg)
    #
    # # tensorboard logger
    # if train_args.use_tfboard:
    #     from tensorboardX import SummaryWriter
    #
    #     tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)
    #
    # # training status for logging
    # training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,
    #                                tblogger if train_args.use_tfboard else None)
    #
    # # total iterations
    # total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch
    # cfg.TRAIN.MAX_ITER = total_iters
    # cfg.TRAIN.GPU_NUM = gpu_num
    #
    # # load model
    # model = MetricDepthModel()
    # output=model(data)

    for i, data in enumerate(train_dataloader):
        break



