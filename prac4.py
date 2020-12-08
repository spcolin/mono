import torch
from lib.models import RD_loss


RD_Loss=RD_loss.RD_loss()


pred=torch.randn(3,1,50,50)

gt=torch.randn(3,1,50,50)

loss=RD_Loss(pred,gt)