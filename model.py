import torch.nn as nn
import torch
import torch.nn.functional as F



class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
    
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1, bias=False, activation='leaky'):
    out=[]
    if bias==False and activation== 'leaky' :
        out.append(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        out.append( nn.BatchNorm2d(out_num),)
        out.append(nn.LeakyReLU())
        
    elif bias==False and activation== 'mish':
        out.append(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        out.append( nn.BatchNorm2d(out_num),)
        out.append(Mish())
    
    else: out.append(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    
    return nn.Sequential(*out)
    



class cspBlock(nn.Module):
    def __init__(self, in_channels, nblocks):
        super(cspBlock, self).__init__()
        out_channels=2*in_channels
        self.layer1 = conv_batch(in_channels, out_channels,stride=2,activation='mish')
        self.split0 = conv_batch(out_channels, out_channels//2, kernel_size=1,padding=0,activation='mish')
        self.split1 = conv_batch(out_channels, out_channels//2, kernel_size=1,padding=0,activation='mish')
        blocks = []
        for i in range(nblocks):
            blocks.append(DarkResidualBlock(out_channels//2))
        self.blocks = nn.Sequential(*blocks)
        self.layer2 = conv_batch(out_channels//2, out_channels//2 ,kernel_size=1,padding=0,activation='mish')
    def forward(self,x):
        x = self.layer1(x)
        split0= self.split0(x)
        split1= self.split1(x)
        blocks = self.blocks(split1)
        x = self.layer2(blocks)
        route = torch.cat([split0, x], dim=1)
        return route


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0,activation='mish')
        self.layer2 = conv_batch(reduced_channels, in_channels,activation='mish')

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

class predict(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(predict, self).__init__()
        self.layer1 = conv_batch(in_channels, in_channels*2)
        self.layer2 = conv_batch(in_channels*2,( num_classes+5)*3, kernel_size=1,padding=0, bias=True)
        self.num_classes = num_classes
    def forward(self, resb):
        out = self.layer1(resb)
        out = self.layer2(out)
        return out.reshape(out.shape[0], 3, self.num_classes + 5, out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
    
            

class cspDarknet53(nn.Module):
    def __init__(self,channels=[32,64,128,256,512], pretrained=False):
        super(cspDarknet53, self).__init__()
    
        self.conv1 = conv_batch(3, 32, activation='mish')
        self.csp_block0 = cspBlock(in_channels=channels[0], nblocks=1)
        self.csp_block1 = cspBlock( in_channels=channels[1], nblocks=2)
        self.csp_block2 = cspBlock( in_channels=channels[2], nblocks=8)
        self.csp_block3 = cspBlock( in_channels=channels[3], nblocks=8)
        self.csp_block4 = cspBlock( in_channels=channels[4], nblocks=4)
        if pretrained==True:
            self.load_pretrained_layers()



    def forward(self, x):
        out = self.conv1(x)
        out = self.csp_block0(out)
        out = self.csp_block1(out)
        out = self.csp_block2(out)
        res8_1=out
        out = self.csp_block3(out)
        res8_2=out
        out = self.csp_block4(out)
        res4_1=out
        return [res8_1, res8_2, res4_1]
    

    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torch.load("cspdarknet53.pth.tar")['state_dict']
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]


        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")



class yolov4(nn.Module):
    def __init__(self,num_classes):
        super(yolov4, self).__init__()
        self.backbone=cspDarknet53()
        for param in self.backbone.parameters():
              	param.requires_grad = True
        self.cbl1=nn.Sequential( conv_batch(1024, 512, kernel_size=1, padding=0),
                                conv_batch(512, 1024),
                                conv_batch(1024, 512, kernel_size=1, padding=0))
        self.spp = SpatialPyramidPooling()

        self.cbl2=nn.Sequential( conv_batch(2048, 512, kernel_size=1, padding=0),
                                conv_batch(512, 1024),
                                conv_batch(1024, 512, kernel_size=1, padding=0))
        self.upsample1=self.upsample(512)
        self.convset1=self.convset5(768,512)
        self.upsample2=self.upsample(256)
        self.convset2=self.convset5(384,256)
        self.cbl3=nn.Sequential( conv_batch(128, 256, kernel_size=3, stride=2))
        self.convset3=self.convset5(512)
        self.cbl4=nn.Sequential(conv_batch(256, 512, kernel_size=3, stride=2))
        self.convset4=self.convset5(1024)
        self.pred8_1=predict(128,num_classes)
        self.pred8_2=predict(256,num_classes)
        self.pred4=predict(512,num_classes)

    def forward(self,image):
        outputs=[]

        out=self.backbone(image)
        out[2]=self.cbl1(out[2])
        out[2]=self.spp(out[2])
        out[2]=self.cbl2(out[2])
        up1=self.upsample1(out[2])
        concat1 = torch.cat([up1, out[1]], dim=1)
        convset1=self.convset1(concat1)
        up2=self.upsample2(convset1)
        concat2 = torch.cat([up2, out[0]], dim=1)
        out[0]=self.convset2(concat2)
        outputs.append(self.pred8_1(out[0]))

        cbl3=self.cbl3(out[0])
        concat3= torch.cat([cbl3, convset1], dim=1)
        out[1]=self.convset3(concat3)
        outputs.append(self.pred8_2(out[1])) 

        cbl4=self.cbl4(out[1])

        concat4= torch.cat([cbl4, out[2]], dim=1)

        out[2]=self.convset4(concat4)
        outputs.append(self.pred4(out[2]))

        return outputs[::-1]
    def upsample(self,in_channels):
        # print(in_channels)
        layer=[]
        layer.append(conv_batch(in_channels,in_channels//2, kernel_size=1, padding=0))
        layer.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layer)
        
    def convset5(self,in_channels,y=None):
        layer=[]
        if y== None:
            y=in_channels
        layer+=conv_batch(in_channels,y//2,kernel_size=1,padding=0)
        layer+=conv_batch(y//2, y)
        layer+=conv_batch(y, y//2,kernel_size=1,padding=0)
        layer+=conv_batch(y//2, y)
        layer+=conv_batch(y, y//2,kernel_size=1,padding=0)
        # print(nn.Sequential(*layer))
        return nn.Sequential(*layer)

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 608
    model = yolov4(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
    #torch.split(x,x.size()[1]//2,dim=1)[0].size()