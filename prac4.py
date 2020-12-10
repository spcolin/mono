"""
test for the depth map magnitude
"""

import torch,torchvision
from PIL import Image

depth_path="/home/colin/p_nyu/home_office_0001_out/1.png"

depth_img=Image.open(depth_path)

print(depth_img.getpixel((0,0)))
print(depth_img.getpixel((0,0))/65535)

tf=torchvision.transforms.ToTensor()(depth_img)

print(tf[0][0][0]/65535)