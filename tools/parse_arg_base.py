import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # ----------------added args---------------------
        # parser.add_argument('--anno_pack_path',  help='Path to package containing annotation json files',default="/home/colin/anno")
        parser.add_argument('--anno_pack_path',  help='Path to package containing annotation json files',default="/home/colin/papercode/VNL_Monocular_Depth_Prediction-master/datasets/NYUDV2/annotations")
        # ----------------added end----------------------

        parser.add_argument('--dataroot',  help='Path to dataset',default="datasets/NYUDV2")
        parser.add_argument('--batchsize', type=int, default=2, help='Batch size')
        parser.add_argument('--cfg_file', default='lib/configs/resnext101_32x4d_nyudv2_class',
                            help='Set model and dataset config files')
        parser.add_argument('--dataset', default='nyudv2', help='the name of dataset,used to specify the dataset')
        parser.add_argument('--load_ckpt', default="/home/ckpt/epoch6_step226000.pth",help='Checkpoint path to load')
        # parser.add_argument('--resume', action='store_true', help='Resume to train')
        parser.add_argument('--resume', default=True, help='Resume to train',type=bool)

        parser.add_argument('--epoch', default=30, type=int, help='Set training epochs')
        parser.add_argument('--start_epoch', default=0, type=int, help='Set training epochs')
        parser.add_argument('--start_step', default=0, type=int, help='Set training steps')
        parser.add_argument('--thread', default=4, type=int, help='Thread for loading data')
        parser.add_argument('--use_tfboard', action='store_true', help='Tensorboard to log training info')
        # not clear
        parser.add_argument('--results_dir', type=str, default='./evaluation', help='Output dir')

        # the param add for DistributedDataParallel
        parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
