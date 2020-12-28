"""
test for the format of .mat file
"""

import scipy.io as scio
import cv2

mat_path="/home/colin/papercode/VNL_Monocular_Depth_Prediction-master/datasets/NYUDV2/train/train.mat"

data=scio.loadmat(mat_path)


# print(data.keys())
# print(data["__header__"])
# print(data["__version__"])
# print(data["__globals__"])


# print(len(data['depths']))
# print(len(data['rgbs']))
# print(len(data['raw_depths']))
# print(len(data['raw_depth_filenames']))
# print(len(data['raw_rgb_filenames']))


depth=data['depths'][0]
print(depth.shape)
depth=depth.transpose((1,0))
print(depth.shape)
print(depth[2])
# print(depth[1][5])
# print(depth[2][5])

# cv2.imshow("rgb",depth)
