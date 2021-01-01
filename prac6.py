"""
test for relative depth loss
"""


import torch
from torchvision import transforms
from PIL import Image
from lib.models.RD_loss import RD_loss
# depth_path="/home/colin/p_nyu/home_office_0001_out/depth/1.png"
# depth_img=Image.open(depth_path)
# depth_tensor=transforms.ToTensor()(depth_img)
# depth_tensor=depth_tensor.unsqueeze(0)/65535.0

simu_tensor=torch.tensor([[5,2,8,2,6],
                          [8,4,1,7,4],
                          [8,2,5,2,4],
                          [3,2,7,3,7],
                          [1,6,2,8,3],
                          [3,7,6,2,1]],dtype=torch.float)
simu_tensor=simu_tensor.unsqueeze(0).unsqueeze(0)



rd_loss=RD_loss()

loss=rd_loss(simu_tensor,simu_tensor)
