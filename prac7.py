"""
test for resnet finetune
"""

import torch,torchvision,collections

model=torchvision.models.resnet18(pretrained=False)

pretrained_path="/home/colin/pretrained/resnet18-5c106cde.pth"


class Net(torch.nn.Module):

    def __init__(self,backbone_model):
        super(Net, self).__init__()
        self.backbone=torch.nn.Sequential(collections.OrderedDict(
            [(k, v) for k, v in backbone_model.named_children()][:-2]
            )
        )

    def load_pretriained_backbone(self,path):
        pretrained_dict=torch.load(path)
        model_dict=self.backbone.state_dict()
        # print(pretrained_dict.keys())
        # print(model_dict.keys())
        to_load_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(to_load_dict)
        self.backbone.load_state_dict(model_dict)


    def forward(self,input):

        output=self.backbone(input)

        return output


net=Net(model)

net.load_pretriained_backbone(pretrained_path)

a=torch.randn(1,3,385,385)


output=net(a)

print(output.shape)


