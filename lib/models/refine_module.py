import torch,torchvision,collections
import torch.nn.functional as F

def conv_layer(in_channels,out_channels,kernel_size=3,stride=1,padding=1):

    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        torch.nn.LeakyReLU(negative_slope=0.01)

    )

class Residual(torch.nn.Module):

    def __init__(self,load_pretrained=True,pretrained_resnet18_path=None):
        super(Residual, self).__init__()

        backbone_model = torchvision.models.resnet18(pretrained=False)

        # -----------------resnet backbone part--------------------
        self.backbone=torch.nn.Sequential(collections.OrderedDict(
            [(k, v) for k, v in backbone_model.named_children()][:-2]
            )
        )

        if load_pretrained==True:
            self.load_pretriained_backbone(pretrained_resnet18_path)

        # adjust the first layer of backbone to have 4 channels(3 for rgb image and 1 for depth map)
        self.backbone.conv1=torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # -----------------backbone end-----------------------------


        # for upsample
        self.up_layer1=conv_layer(512,256)
        self.up_layer2=conv_layer(256,128)

        self.up_layer3=conv_layer(132,64)
        self.up_layer4=conv_layer(64,32)

        self.up_layer5 = conv_layer(36, 16)
        self.up_layer6=torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # prediction layer
        self.predict=torch.nn.Sigmoid()


    def load_pretriained_backbone(self,path):
        pretrained_dict=torch.load(path)
        model_dict=self.backbone.state_dict()
        # print(pretrained_dict.keys())
        # print(model_dict.keys())
        to_load_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(to_load_dict)
        self.backbone.load_state_dict(model_dict)

        print("-----pretrained resnet18 loaded-----")


    def forward(self,input):

        feature=self.backbone(input)    #B*512*13*13

        up1=F.interpolate(feature,scale_factor=2,mode='bilinear',align_corners=True)    #B*512*26*26
        up1=self.up_layer1(up1) #B*256*26*26
        up1=F.interpolate(up1,scale_factor=2,mode='bilinear',align_corners=True)    #B*256*52*52
        up1=self.up_layer2(up1) #B*128*52*52

        cat1=torch.cat([up1,F.interpolate(input,size=up1.shape[2:])],1) #B*132*52*52

        up2 = F.interpolate(cat1, scale_factor=2, mode='bilinear', align_corners=True)  #B*132*104*104
        up2 = self.up_layer3(up2)   #B*64*104*104
        up2 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)   #B*64*208*208
        up2 = self.up_layer4(up2)   #B*32*208*208

        cat2=torch.cat([up2,F.interpolate(input,size=up2.shape[2:])],1) #B*36*208*208

        up = F.interpolate(cat2, scale_factor=2, mode='bilinear', align_corners=True)   #B*32*416*416
        up = self.up_layer5(up) #B*16*416*416
        up = F.interpolate(up,size=(385,385), mode='bilinear', align_corners=True)  #B*16*385*385
        prediction=self.up_layer6(up)   #B*1*385*385
        prediction=self.predict(prediction)*0.1 #B*1*385*385


        return prediction


pretrained_path="/home/colin/pretrained/resnet18-5c106cde.pth"

residual_module=Residual(pretrained_resnet18_path=pretrained_path)

input=torch.randn(2,4,385,385)

output=residual_module(input)

print(output.shape)




