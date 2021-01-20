import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging
import time
from functools import wraps
import lib.core.config


def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(1)
        return ret
    return wrapper



logger = setup_logging(__name__)


class NYUDV2Dataset():
    def initialize(self, opt):

        self.cfg=lib.core.config.cfg

        self.opt = opt
        self.root = opt.dataroot
        self.depth_normalize = 60000.
        # self.dir_anno = cfg.ROOT_DIR+opt.dataroot+'/annotations/'+ opt.phase_anno + '_annotations.json'
        # self.dir_anno = os.path.join(cfg.ROOT_DIR, opt.dataroot, 'annotations', opt.phase_anno + '_annotations.json')
        self.dir_anno = os.path.join(opt.anno_pack_path, opt.phase_anno + '_annotations.json')
        # print("*****")
        # print(cfg.ROOT_DIR)
        # print(opt.dataroot)
        # print(opt.phase_anno)
        # print(self.dir_anno)
        # print("-----")
        self.A_paths, self.B_paths, self.AB_anno = self.getData()
        self.data_size = len(self.AB_anno)
        self.uniform_size = (480, 640)

    def getData(self):

        cfg=self.cfg
        # print(self.opt.phase_anno)
        # print("type search:", type(cfg.DATASET.DEPTH_BIN_INTERVAL))

        # print(self.dir_anno)
        load_f=open(self.dir_anno, 'r')

        AB_anno = json.load(load_f)
        # print("*****")
        # print(AB_anno[0].keys())
        # print(AB_anno[0])
        # print("-----")

        if 'dir_AB' in AB_anno[0].keys():
            self.dir_AB = os.path.join(cfg.ROOT_DIR, self.opt.dataroot, self.opt.phase_anno, AB_anno[0]['dir_AB'])
            # print(self.dir_AB)
            AB = sio.loadmat(self.dir_AB)
            # print(AB.keys())
            self.A = AB['rgbs']
            self.B = AB['depths']
            self.depth_normalize = 10.0
            print("mat way")

        else:

            print("image way")

            self.A = None
            self.B = None

            #scale of depth
            self.depth_normalize=65535

        A_list = [os.path.join(cfg.ROOT_DIR, self.opt.dataroot, self.opt.phase_anno, AB_anno[i]['rgb_path']) for i in range(len(AB_anno))]
        B_list = [os.path.join(cfg.ROOT_DIR, self.opt.dataroot, self.opt.phase_anno, AB_anno[i]['depth_path']) for i in range(len(AB_anno))]
        logger.info('Loaded NYUDV2 data!')


        # print("*****")
        # print(A_list[0])
        # print(B_list[0])
        # print("-----")

        return A_list, B_list, AB_anno

    @loop_until_success
    def __getitem__(self, anno_index):
        # print("-------------------------in getitem-------------------------")
        # print_configs(lib.core.config.cfg)
        # print("-------------------------in getitem-------------------------")

        data = self.online_aug(anno_index)
        return data

    def online_aug(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """


        # A:rgb image B:ground truth depth
        A_path = self.A_paths[anno_index]
        B_path = self.B_paths[anno_index]

        if self.A is None:
            A = cv2.imread(A_path)  # bgr, H*W*C
            B = cv2.imread(B_path, -1) / self.depth_normalize  # the max depth is 10m
        else:
            A = self.A[anno_index]  # C*W*H
            B = self.B[anno_index] / self.depth_normalize # the max depth is 10m
            # print("*********************")
            # print(A.shape)
            # print(B.shape)
            # print("---------------------")
            A = A.transpose((2, 1, 0))  # H * W * C
            B = B.transpose((1, 0))  # H * W
            A = A[:, :, ::-1].copy() #rgb -> bgr

        # print("*********************")
        # print("image shape:",A.shape)
        # print("depth map shape:",B.shape)
        # print("---------------------")
        # print("*********************")
        # print("depth value:",B[50][50])
        # print("---------------------")


        flip_flg, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop()

        A_resize = self.flip_pad_reshape_crop(A, flip_flg, crop_size, pad, 128)
        B_resize = self.flip_pad_reshape_crop(B, flip_flg, crop_size, pad, -1)

        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        A_resize = A_resize[::-1, :, :]

        # to torch, normalize
        # print("img shape:",A_resize.shape)
        A_resize = self.scale_torch(A_resize, 255.)
        B_resize = self.scale_torch(B_resize, resize_ratio)


        B_bins = self.depth_to_bins(B_resize)
        invalid_side = [int(pad[0] * resize_ratio), 0, 0, 0]

        # print("*****")
        # print(B_bins.shape)
        # print("-----")

        data = {'A': A_resize, 'B': B_resize, 'A_raw': A, 'B_raw': B, 'B_bins': B_bins, 'A_paths': A_path,
                'B_paths': B_path, 'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def set_flip_pad_reshape_crop(self):
        """
        Set flip, padding, reshaping, and cropping factors for the image.
        :return:
        """
        cfg = self.cfg

        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        raw_size = np.array([cfg.DATASET.CROP_SIZE[1], 416, 448, 480, 512, 544, 576, 608, 640])

        size_index = np.random.randint(0, 9) if 'train' in self.opt.phase else 8

        # pad
        pad_height = raw_size[size_index] - self.uniform_size[0] if raw_size[size_index] > self.uniform_size[0]\
                    else 0

        if pad_height!=0:
            pad_up=np.random.randint(0,pad_height)
            pad_down=pad_height-pad_up
            pad = [pad_up, pad_down, 0, 0]  # [up, down, left, right]
        else:
            pad = [pad_height, 0, 0, 0]  # [up, down, left, right]
            # pad = [pad_up, pad_down, 0, 0]  # [up, down, left, right]

        # crop
        crop_height = raw_size[size_index]
        crop_width = raw_size[size_index]
        start_x = np.random.randint(0, int(self.uniform_size[1] - crop_width)+1)
        # start_y = 0 if pad_height != 0 else np.random.randint(0,
        #         int(self.uniform_size[0] - crop_height) + 1)
        start_y = np.random.randint(0,pad_height) if pad_height != 0 \
            else np.random.randint(0,int(self.uniform_size[0] - crop_height) + 1)

        crop_size = [start_x, start_y, crop_height, crop_width]

        resize_ratio = float(cfg.DATASET.CROP_SIZE[1] / crop_width)

        return flip_flg, crop_size, pad, resize_ratio

    def flip_pad_reshape_crop(self, img, flip, crop_size, pad, pad_value=0):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        cfg = self.cfg


        # print(img.shape)
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # print("image shape:",img.shape)
        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                       constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))


        # Crop the padded image
        img_crop = img_pad[crop_size[1]:crop_size[1] + crop_size[2], crop_size[0]:crop_size[0] + crop_size[3]]


        # Resize the raw image
        img_resize = cv2.resize(img_crop, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        return img_resize

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        cfg = self.cfg

        # print("depth type:",type(depth))
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX


        # calculate the bin for a certain depth value
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        # print("bins shape:",bins.shape)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        # print("*****")
        # print("depth value:",depth[0,100,100])
        # print("min log depth:",cfg.DATASET.DEPTH_MIN_LOG)
        # print("bin interval:",cfg.DATASET.DEPTH_BIN_INTERVAL)
        # print("depth bin:",bins[0,100,100])
        # print("-----")



        return bins

    def scale_torch(self, img, scale):
        """
        Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        cfg=self.cfg
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
        else:
            img = transforms.Normalize((0,), (1,))(img)
        return img

    def __len__(self):
        return self.data_size

    def name(self):
        return 'NYUDV2'

