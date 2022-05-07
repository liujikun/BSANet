import torch
import torch.nn as nn
from network_module import *
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import math
# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
#-----------------------------------------
#         _DCR_block
# ----------------------------------------
class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(int(channel_in/2.), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(int(channel_in), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv_1(x)))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.bn2(self.conv_2(conc)))
        return out

class Block_of_DMT0(nn.Module):
    def __init__(self):
        super(Block_of_DMT0,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(64, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output 

class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(128, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output 
 
class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2,self).__init__()
 
        #DMT1
        self.conv2_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(256, affine=True)
        self.relu2_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        return output 


 
class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3,self).__init__()
 
        #DMT1
        self.conv3_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(256, affine=True)
        self.relu3_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        return output 



class Block_of_DMT4(nn.Module):
    def __init__(self):
        super(Block_of_DMT4,self).__init__()
 
        #DMT1
        self.conv4_1=nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(1024, affine=True)
        self.relu4_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        return output

class Block_of_DMT01(nn.Module):
    def __init__(self):
        super(Block_of_DMT01,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(32, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        # output = self.relu1_1(self.conv1_1(x))
        return output 

class Block_of_DMT11(nn.Module):
    def __init__(self):
        super(Block_of_DMT11,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(64, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        # output = self.relu1_1(self.conv1_1(x))
        return output 
 
class Block_of_DMT21(nn.Module):
    def __init__(self):
        super(Block_of_DMT21,self).__init__()
 
        #DMT1
        self.conv2_1=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(128, affine=True)
        self.relu2_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        # output = self.relu2_1(self.conv2_1(x))
        return output 


 
class Block_of_DMT31(nn.Module):
    def __init__(self):
        super(Block_of_DMT31,self).__init__()
 
        #DMT1
        self.conv3_1=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(128, affine=True)
        self.relu3_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        # output = self.relu3_1(self.conv3_1(x))
        
        return output 



class Block_of_DMT41(nn.Module):
    def __init__(self):
        super(Block_of_DMT41,self).__init__()
 
        #DMT1
        self.conv4_1=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(512, affine=True)
        self.relu4_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        # output = self.relu4_1(self.conv4_1(x))
        return output

class Generator_segdila(nn.Module):
    def __init__(self, args,num_classes):
        super(Generator_segdila, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E00 = Conv2dLayer(in_channels = 32,  out_channels = 32, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E000 = Conv2dLayer(in_channels = 32,  out_channels = 32, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)

        self.E1 = Conv2dLayer(in_channels = 128,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E11 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E111 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)

        self.E2 = Conv2dLayer(in_channels = 256, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E22 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E222 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.E3 = Conv2dLayer(in_channels = 512, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E33 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E333 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.E4 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E44 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E444 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UE0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE00 = Conv2dLayer(in_channels = 32,  out_channels = 32, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE000 = Conv2dLayer(in_channels = 32,  out_channels = 32, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)

        self.UE1 = Conv2dLayer(in_channels = 32,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE11 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE111 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE2 = Conv2dLayer(in_channels = 64, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE22 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE222 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UE3 = Conv2dLayer(in_channels = 128, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE33 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE333 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UE4 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE44 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE444 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UD1 = Conv2dLayer(in_channels = 512,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD11 = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD111 = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)

        self.UD2 = Conv2dLayer(in_channels = 256, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD22 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD222 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UD3 = Conv2dLayer(in_channels = 128, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD33 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD333 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.UD4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD444 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(2)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D2 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D3 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = 'relu')
        self.D11 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D111 = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)

        self.D22 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D222 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D33 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D33 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D333 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 3, dilation = 3,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.D44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = args.dila, dilation = args.dila, pad_type = args.pad, norm = args.norm, activation = 'relu')
        self.D444 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 3, dilation = 3, pad_type = args.pad, norm = args.norm, activation = 'relu')
        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')
        self.S1 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = args.pad, activation = 'none', norm = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        #DMT1
        E0 = self.E0(x)
        E0 = self.E00(E0)
        E0 = self.E000(E0)
        DMT1_yl,DMT1_yh = self.DWT(E0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E1 = self.E1(DMT1)
        E1 = self.E11(E1)    # channel = 128       
        E1 = self.E111(E1)    # channel = 128       

        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2 = self.E2(DMT2)
        E2 = self.E22(E2)    # channel = 256
        E2 = self.E222(E2)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3 = self.E3(DMT3)
        E3 = self.E33(E3)     #channel = 256
        E3 = self.E333(E3)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3)
        DMT4 = self._transformer(DMT4_yl, DMT4_yh)
        E4 = self.E4(DMT4)
        E4 = self.E44(E4)     #channel = 256
        E4_out = self.E444(E4)     #channel = 256
        
        # UE
        UE0 = self.UE0(x)
        UE0 = self.UE00(UE0)
        UE0 = self.UE000(UE0)
        UE1 = self.down(UE0)
        UE1 = self.UE1(UE1)
        UE1 = self.UE11(UE1)
        UE1 = self.UE111(UE1)
        UE2 = self.down(UE1)
        UE2 = self.UE2(UE2)
        UE2 = self.UE22(UE2)
        UE2 = self.UE222(UE2)          
        UE3 = self.down(UE2)
        UE3 = self.UE3(UE3)
        UE3 = self.UE33(UE3)
        UE3 = self.UE333(UE3)
        UE4 = self.down(UE3)
        UE4 = self.UE4(UE4)
        UE4 = self.UE44(UE4)
        UE4_out = self.UE444(UE4)

        # UD
        UE4 = self.up(UE4_out)
        UD1=torch.cat((UE4, UE3), 1)
        UD1 = self.UD1(UD1)
        UD1 = self.UD11(UD1)
        UD1 = self.UD111(UD1)

        UD1 = self.up(UD1)
        UD2 = torch.cat((UD1, UE2), 1)
        UD2 = self.UD2(UD2)
        UD2 = self.UD22(UD2)
        UD2 = self.UD222(UD2)
        UD2_MIDDLE = self.UD222(UD2)         
         
        UD2 = self.up(UD2_MIDDLE)
        UD3 = torch.cat((UD2, UE1), 1)
        UD3 = self.UD3(UD3)
        UD3 = self.UD33(UD3)
        UD3 = self.UD333(UD3)

        UD3 = self.up(UD3)
        UD4 = torch.cat((UD3, E0), 1)
        UD4 = self.UD4(UD4)
        UD4 = self.UD44(UD4)
        UD4 = self.UD444(UD4)
        #IDMT4
        
        D1=self._Itransformer(E4_out)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1=self.D1(D1)
        D1=self.D11(D1)
        D1=self.D111(D1)
        #IDMT3
        
        D2=self._Itransformer(D1)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2=self.D2(D2)
        D2=self.D22(D2)
        D2=self.D222(D2)
        #IDMT2
        D3=self._Itransformer(D2)
        IDMT2=self.IDWT(D3)   #128
        D3=torch.cat((IDMT2, E1), 1)  #256
        D3=self.D3(D3)  
        D3=self.D33(D3)  #256
        D3=self.D333(D3)  #256
        #IDMT1
        D4 = self._Itransformer(D3)
        IDMT1 = self.IDWT(D4)   #64
        D4=torch.cat((IDMT1, E0), 1)  #128
        D4 = self.D4(D4)
        D4 = self.D44(D4)
        D4 = self.D444(D4)

        # merge
        merge = torch.cat((D4, UD4), 1) 
        merge1 = self.S1(merge)

        x = self.out_conv(merge1)
        # return x, UE4_out,E4_out,merge
        return x, UE4_out,E4_out,merge,UD2_MIDDLE,D2
        # return x, UD2_MIDDLE,D2,merge
class Wave_Generator(nn.Module):
    def __init__(self, args,num_classes):
        super(Wave_Generator, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E1 = Conv2dLayer(in_channels = 128,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E2 = Conv2dLayer(in_channels = 256, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E3 = Conv2dLayer(in_channels = 512, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E4 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.blockDMT0 = self.make_layer(Block_of_DMT01,1)
        self.blockDMT1 = self.make_layer(Block_of_DMT11,1)
        self.blockDMT2 = self.make_layer(Block_of_DMT21,1)
        self.blockDMT3 = self.make_layer(Block_of_DMT31,1)
        self.blockDMT4 = self.make_layer(Block_of_DMT41,1)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D2 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D3 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)
        self.D11 = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D22 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D33 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)

        self.E0d = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E1d = Conv2dLayer(in_channels = 128,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E2d = Conv2dLayer(in_channels = 256, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E3d = Conv2dLayer(in_channels = 512, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E4d = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.blockDMT0d = self.make_layer(Block_of_DMT01,1)
        self.blockDMT1d = self.make_layer(Block_of_DMT11,1)
        self.blockDMT2d = self.make_layer(Block_of_DMT21,1)
        self.blockDMT3d = self.make_layer(Block_of_DMT31,1)
        self.blockDMT4d = self.make_layer(Block_of_DMT41,1)
        # Decoder
        self.D1d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D2d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D3d = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D4d = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)
        self.D11d = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D22d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D33d = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D44d = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)
        

        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')
        self.S1 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = args.pad, activation = 'none', norm = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        #DMT1
        E0 = self.E0(x)
        E0 = self.blockDMT0(E0)
        DMT1_yl,DMT1_yh = self.DWT(E0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E1 = self.E1(DMT1)
        E1 = self.blockDMT1(E1)    # channel = 128       

        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2 = self.E2(DMT2)
        E2 = self.blockDMT2(E2)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3 = self.E3(DMT3)
        E3 = self.blockDMT3(E3)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3)
        DMT4 = self._transformer(DMT4_yl, DMT4_yh)
        E4_out = self.E4(DMT4)
        E4 = self.blockDMT4(E4_out)     #channel = 256
        
        D1=self._Itransformer(E4)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1=self.D1(D1)
        D1=self.D11(D1)
        #IDMT3
        
        D2=self._Itransformer(D1)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2=self.D2(D2)
        D2=self.D22(D2)
        #IDMT2
        D3=self._Itransformer(D2)
        IDMT2=self.IDWT(D3)   #128
        D3=torch.cat((IDMT2, E1), 1)  #256
        D3=self.D3(D3)  
        D3=self.D33(D3)  #256
        #IDMT1
        D4 = self._Itransformer(D3)
        IDMT1 = self.IDWT(D4)   #64
        D4=torch.cat((IDMT1, E0), 1)  #128
        D4 = self.D4(D4)
        D4 = self.D44(D4)

        E1d = self.E1d(DMT1)
        E1d = self.blockDMT1d(E1d)    # channel = 128       
        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1d)
        DMT2d = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2d = self.E2d(DMT2d)
        E2d = self.blockDMT2d(E2d)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2d)
        DMT3d = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3d = self.E3d(DMT3d)
        E3d = self.blockDMT3d(E3d)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3d)
        DMT4d = self._transformer(DMT4_yl, DMT4_yh)
        E4_outd = self.E4d(DMT4d)
        E4d = self.blockDMT4(E4_outd)     #channel = 256
        
        D1d=self._Itransformer(E4d)
        IDMT4d=self.IDWT(D1d)
        D1d=torch.cat((IDMT4d, E3d), 1)
        D1d=self.D1d(D1d)
        D1d=self.D11d(D1d)
        #IDMT3
        
        D2d=self._Itransformer(D1d)
        IDMT3d=self.IDWT(D2d)
        D2d=torch.cat((IDMT3d, E2d), 1)
        D2d=self.D2d(D2d)
        D2d=self.D22d(D2d)
        #IDMT2
        D3d=self._Itransformer(D2d)
        IDMT2d=self.IDWT(D3d)   #128
        D3d=torch.cat((IDMT2d, E1d), 1)  #256
        D3d=self.D3d(D3d)  
        D3d=self.D33d(D3d)  #256
        #IDMT1
        D4d = self._Itransformer(D3d)
        IDMT1d = self.IDWT(D4d)   #64
        D4d=torch.cat((IDMT1d, E0), 1)  #128
        D4d = self.D4d(D4d)
        D4d = self.D44d(D4d)

        # merge
        merge = torch.cat((D4, D4d), 1) 
        merge1 = self.S1(merge)

        x = self.out_conv(merge1)
        return x, D4,D4d,merge,D2d,D2
        # return x, UE4_out,E4_out,merge,UD2_MIDDLE,D2
        # return x, UD2_MIDDLE,D2,merge

class Generator(nn.Module):
    def __init__(self, args,num_classes):
        super(Generator, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E1 = Conv2dLayer(in_channels = 128,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E2 = Conv2dLayer(in_channels = 256, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E3 = Conv2dLayer(in_channels = 512, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E4 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        # self.UE0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        # self.UE00 = Conv2dLayer(in_channels = 32,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE1 = Conv2dLayer(in_channels = 32,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE11 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE2 = Conv2dLayer(in_channels = 64, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE22 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE3 = Conv2dLayer(in_channels = 128, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE33 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE4 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE44 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD1 = Conv2dLayer(in_channels = 512,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD11 = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD2 = Conv2dLayer(in_channels = 256, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD22 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD3 = Conv2dLayer(in_channels = 128, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD33 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(2)
        self.blockDMT0 = self.make_layer(Block_of_DMT01,1)
        self.blockDMT1 = self.make_layer(Block_of_DMT11,1)
        self.blockDMT2 = self.make_layer(Block_of_DMT21,1)
        self.blockDMT3 = self.make_layer(Block_of_DMT31,1)
        self.blockDMT4 = self.make_layer(Block_of_DMT41,1)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D2 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D3 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)
        self.D11 = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D22 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D33 = Conv2dLayer(in_channels = 128, out_channels = 128,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = args.act, norm = args.norm)
        self.D44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = args.act)
        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')
        self.S1 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = args.pad, activation = 'none', norm = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        #DMT1
        E0 = self.E0(x)
        E0 = self.blockDMT0(E0)
        DMT1_yl,DMT1_yh = self.DWT(E0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E1 = self.E1(DMT1)
        E1 = self.blockDMT1(E1)    # channel = 128       

        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2 = self.E2(DMT2)
        E2 = self.blockDMT2(E2)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3 = self.E3(DMT3)
        E3 = self.blockDMT3(E3)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3)
        DMT4 = self._transformer(DMT4_yl, DMT4_yh)
        E4_out = self.E4(DMT4)
        E4 = self.blockDMT4(E4_out)     #channel = 256
        
        # UE
        # UE0 = self.UE0(x)
        # UE0 = self.UE00(UE0)
        UE1 = self.down(E0)
        UE1 = self.UE1(UE1)
        UE1 = self.UE11(UE1)
        UE2 = self.down(UE1)
        UE2 = self.UE2(UE2)
        UE2 = self.UE22(UE2)         
        UE3 = self.down(UE2)
        UE3 = self.UE3(UE3)
        UE3 = self.UE33(UE3)
        UE4 = self.down(UE3)
        UE4_out = self.UE4(UE4)
        UE4 = self.UE44(UE4_out)

        # UD
        UE4 = self.up(UE4)
        UD1=torch.cat((UE4, UE3), 1)
        UD1 = self.UD1(UD1)
        UD1 = self.UD11(UD1)
        UD1 = self.up(UD1)
        UD2 = torch.cat((UD1, UE2), 1)
        UD2 = self.UD2(UD2)
        UD2_MIDDLE = self.UD22(UD2)         
        UD2 = self.up(UD2_MIDDLE)
        UD3 = torch.cat((UD2, UE1), 1)
        UD3 = self.UD3(UD3)
        UD3 = self.UD33(UD3)
        UD3 = self.up(UD3)
        UD4 = torch.cat((UD3, E0), 1)
        UD4 = self.UD4(UD4)
        UD4 = self.UD44(UD4)
        #IDMT4
        
        D1=self._Itransformer(E4)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1=self.D1(D1)
        D1=self.D11(D1)
        #IDMT3
        
        D2=self._Itransformer(D1)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2=self.D2(D2)
        D2=self.D22(D2)
        #IDMT2
        D3=self._Itransformer(D2)
        IDMT2=self.IDWT(D3)   #128
        D3=torch.cat((IDMT2, E1), 1)  #256
        D3=self.D3(D3)  
        D3=self.D33(D3)  #256
        #IDMT1
        D4_out = self._Itransformer(D3)
        IDMT1 = self.IDWT(D4_out)   #64
        D4=torch.cat((IDMT1, E0), 1)  #128
        D4 = self.D4(D4)
        D4 = self.D44(D4)

        # merge
        merge = torch.cat((D4, UD4), 1) 
        merge1 = self.S1(merge)

        x = self.out_conv(merge1)
        return x, D4,UD4,merge,E4_out,UE4_out
        # return x, UE4_out,E4_out,merge,UD2_MIDDLE,D2
        # return x, UD2_MIDDLE,D2,merge

class doubleUGenerator(nn.Module):
    def __init__(self, args,num_classes):
        super(doubleUGenerator, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE1 = Conv2dLayer(in_channels = 32,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE11 = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE2 = Conv2dLayer(in_channels = 64, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE22 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE3 = Conv2dLayer(in_channels = 128, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE33 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE4 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE44 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD1 = Conv2dLayer(in_channels = 512,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD11 = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD2 = Conv2dLayer(in_channels = 256, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD22 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD3 = Conv2dLayer(in_channels = 128, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD33 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD4 = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD44 = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        self.blockDMT0 = self.make_layer(Block_of_DMT01,1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(2)

        self.UE1d = Conv2dLayer(in_channels = 32,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE11d = Conv2dLayer(in_channels = 64,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE2d = Conv2dLayer(in_channels = 64, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE22d = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE3d = Conv2dLayer(in_channels = 128, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE33d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE4d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE44d = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD1d = Conv2dLayer(in_channels = 512,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD11d = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD2d = Conv2dLayer(in_channels = 256, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD22d = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD3d = Conv2dLayer(in_channels = 128, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD33d = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD4d = Conv2dLayer(in_channels = 64, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD44d = Conv2dLayer(in_channels = 32, out_channels = 32, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)

        # Decoder
        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')
        self.S1 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = args.pad, activation = 'none', norm = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        E0 = self.E0(x)
        E0 = self.blockDMT0(E0)
        UE1 = self.down(E0)
        UE1 = self.UE1(UE1)
        UE1 = self.UE11(UE1)
        UE2 = self.down(UE1)
        UE2 = self.UE2(UE2)
        UE2 = self.UE22(UE2)         
        UE3 = self.down(UE2)
        UE3 = self.UE3(UE3)
        UE3 = self.UE33(UE3)
        UE4 = self.down(UE3)
        UE4_out = self.UE4(UE4)
        UE4 = self.UE44(UE4_out)

        # UD
        UE4 = self.up(UE4)
        UD1=torch.cat((UE4, UE3), 1)
        UD1 = self.UD1(UD1)
        UD1 = self.UD11(UD1)
        UD1 = self.up(UD1)
        UD2 = torch.cat((UD1, UE2), 1)
        UD2 = self.UD2(UD2)
        UD2_MIDDLE = self.UD22(UD2)         
        UD2 = self.up(UD2_MIDDLE)
        UD3 = torch.cat((UD2, UE1), 1)
        UD3 = self.UD3(UD3)
        UD3 = self.UD33(UD3)
        UD3 = self.up(UD3)
        UD4 = torch.cat((UD3, E0), 1)
        UD4 = self.UD4(UD4)
        UD4 = self.UD44(UD4)
        
        UE1d = self.down(E0)
        UE1d = self.UE1d(UE1d)
        UE1d = self.UE11d(UE1d)
        UE2d = self.down(UE1d)
        UE2d = self.UE2d(UE2d)
        UE2d = self.UE22d(UE2d)         
        UE3d = self.down(UE2d)
        UE3d = self.UE3d(UE3d)
        UE3d = self.UE33d(UE3d)
        UE4d = self.down(UE3d)
        UE4_outd = self.UE4d(UE4d)
        UE4d = self.UE44d(UE4_outd)

        # UD
        UE4d = self.up(UE4d)
        UD1d=torch.cat((UE4d, UE3d), 1)
        UD1d = self.UD1d(UD1d)
        UD1d = self.UD11d(UD1d)
        UD1d = self.up(UD1d)
        UD2d = torch.cat((UD1d, UE2d), 1)
        UD2d = self.UD2d(UD2d)
        UD2_MIDDLEd = self.UD22d(UD2d)         
        UD2d = self.up(UD2_MIDDLEd)
        UD3d = torch.cat((UD2d, UE1d), 1)
        UD3d = self.UD3d(UD3d)
        UD3d = self.UD33d(UD3d)
        UD3d = self.up(UD3d)
        UD4d = torch.cat((UD3d, E0), 1)
        UD4d = self.UD4d(UD4d)
        UD4d = self.UD44d(UD4d)
        # merge
        merge = torch.cat((UD4d, UD4), 1) 
        merge1 = self.S1(merge)

        x = self.out_conv(merge1)
        return x, UD4,UD4d,merge,UD2_MIDDLE,UD2_MIDDLEd

class Generator1(nn.Module):
    def __init__(self, args,num_classes):
        super(Generator1, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E1 = Conv2dLayer(in_channels = 256,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.E2 = Conv2dLayer(in_channels = 512, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E3 = Conv2dLayer(in_channels = 1024, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.E4 = Conv2dLayer(in_channels = 1024, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        # Bottle neck
        # self.BottleNeck = nn.Sequential(
        #     ResConv2dLayer(256, 3, 1, 1, pad_type = args.pad, norm = args.norm),
        #     ResConv2dLayer(256, 3, 1, 1, pad_type = args.pad, norm = args.norm),
        #     ResConv2dLayer(256, 3, 1, 1, pad_type = args.pad, norm = args.norm),
        #     ResConv2dLayer(256, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        # )
        self.blockDMT0 = self.make_layer(Block_of_DMT0,1)
        self.blockDMT1 = self.make_layer(Block_of_DMT1,1)
        self.blockDMT2 = self.make_layer(Block_of_DMT2,1)
        self.blockDMT3 = self.make_layer(Block_of_DMT3,1)
        self.blockDMT4 = self.make_layer(Block_of_DMT4,1)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D2 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D3 = Conv2dLayer(in_channels = 256, out_channels = 256,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D4 = Conv2dLayer(in_channels = 128, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = 'prelu')
        self.D11 = Conv2dLayer(in_channels = 512, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D22 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D33 = Conv2dLayer(in_channels = 256, out_channels = 256,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'prelu', norm = args.norm)
        self.D44 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = args.pad, norm = args.norm, activation = 'prelu')
        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=1, stride = 1, padding = 0, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        #DMT1
        E0 = self.E0(x)
        E0 = self.blockDMT0(E0)
        DMT1_yl,DMT1_yh = self.DWT(E0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E1 = self.E1(DMT1)
        E1 = self.blockDMT1(E1)    # channel = 128       

        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2 = self.E2(DMT2)
        E2 = self.blockDMT2(E2)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3 = self.E3(DMT3)
        E3 = self.blockDMT3(E3)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3)
        DMT4 = self._transformer(DMT4_yl, DMT4_yh)
        # print(DMT4.shape)
        E4 = self.E4(DMT4)
        E4 = self.blockDMT4(E4)     #channel = 256
        # E4 = self.BottleNeck(E4)

        #IDMT4
        
        D1=self._Itransformer(E4)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1=self.D1(D1)
        D1=self.D11(D1)
        #IDMT3
        
        D2=self._Itransformer(D1)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2=self.D2(D2)
        D2=self.D22(D2)
        #IDMT2
        D3=self._Itransformer(D2)
        IDMT2=self.IDWT(D3)   #128
        D3=torch.cat((IDMT2, E1), 1)  #256
        D3=self.D3(D3)  
        D3=self.D33(D3)  #256
        #IDMT1
        D4 = self._Itransformer(D3)
        IDMT1 = self.IDWT(D4)   #64
        D4=torch.cat((IDMT1, E0), 1)  #128
        D4 = self.D4(D4)
        D4 = self.D44(D4)
        x = self.out_conv(D4)
        return x , D4

class DSWN(nn.Module):
    def __init__(self, args,num_classes):
        super(DSWN, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(in_channels = args.in_channels,  out_channels = 80, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu', norm = 'bn')
        self.E2 = Conv2dLayer(in_channels = 3*4, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu',norm = 'bn')
        self.E3 = Conv2dLayer(in_channels = 3*4*4, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu',norm = 'bn')
        self.E4 = Conv2dLayer(in_channels = 3*4*16, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu',norm = 'bn')

        self.blockDMT11 = self.make_layer(_DCR_block, 160)
        # self.blockDMT12 = self.make_layer(_DCR_block, 160)
        self.blockDMT21 = self.make_layer(_DCR_block, 256)
        # self.blockDMT22 = self.make_layer(_DCR_block, 256)
        self.blockDMT31 = self.make_layer(_DCR_block, 256)
        # self.blockDMT32 = self.make_layer(_DCR_block, 256)
        self.blockDMT41 = self.make_layer(_DCR_block, 128)
        # self.blockDMT42 = self.make_layer(_DCR_block, 128)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 128, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu', norm = 'bn')
        self.D2 = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu', norm = 'bn')
        self.D3 = Conv2dLayer(in_channels = 256, out_channels = 320,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = 'zero', activation = 'prelu', norm = 'bn')
        self.D4 = Conv2dLayer(in_channels = 160, out_channels = num_classes, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', norm = 'none', activation = 'none')
        # channel shuffle
        self.S1 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = 'zero', activation = 'none', norm = 'none')
        self.S2 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = 'zero', activation = 'none', norm = 'none')
        self.S3 = Conv2dLayer(in_channels = 160, out_channels = 160, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = 'zero', activation = 'none', norm = 'none')

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh
    def forward(self, x):
        E1 = self.E1(x)
        DMT1_yl,DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E2 = self.E2(DMT1)
        
        DMT2_yl, DMT2_yh = self.DWT(DMT1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        E3 = self.E3(DMT2)

        DMT3_yl, DMT3_yh = self.DWT(DMT2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        E4 = self.E4(DMT3)

        E4 = self.blockDMT41(E4)
        x1 =self.D1(E4)
        D1=self._Itransformer(x1)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1 = self.S1(D1)
        D2=self.blockDMT31(D1)
        x2=self.D2(D2)

        D2=self._Itransformer(x2)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2 = self.S2(D2)
        D3=self.blockDMT21(D2)

        x3=self.D3(D3)

        D3=self._Itransformer(x3)
        IDMT2=self.IDWT(D3)
        D3=torch.cat((IDMT2, E1), 1)
        
        # res branch
        D4 = self.S3(D3)
        D4= self.blockDMT11(D4)
        # x4 = self.blockDMT12(D4)
        output = self.D4(D4)
        

        return output


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# VGG style Discriminator with input size 128*128
def conv_block(in_nc, out_nc, kernel_size, stride = 1, dilation = 1, groups = 1, bias = True, \
               pad_type = 'zero', norm_type = None, act_type = 'relu', mode = 'CNA'):
    
    #Conv layer with padding, normalization, activation
    #mode: CNA --> Conv -> Norm -> Act
    #    NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    # cc = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=stride, padding=padding, \
    #         dilation=dilation, bias=bias, groups=groups)
    # aa = act(act_type) if act_type else None
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
    
class VGG128_Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type = 'batch', act_type = 'leakyrelu', mode = 'CNA'):
        super(VGG128_Discriminator, self).__init__()
        # features
        # hxw, c
        # in_nc = 3
        # base_nf = 64
        # 128, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 32, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 16, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 8, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 4, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)
        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70_1(nn.Module):
    def __init__(self, args):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(512, args.start_channels, 1, 1, 0, pad_type = args.pad, norm = 'none'),
            Conv2dLayer(args.start_channels, args.start_channels, 7, 1, 3, pad_type = args.pad, norm = args.norm)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(args.start_channels , args.start_channels * 2, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 2, args.start_channels * 2, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(args.start_channels * 2, args.start_channels * 4, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 4, args.start_channels * 4, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(args.start_channels * 4, args.start_channels * 8, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 8, args.start_channels * 8, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(args.start_channels * 8, args.start_channels * 4, 4, 1, 1, pad_type = args.pad, norm = args.norm)
        self.final2 = Conv2dLayer(args.start_channels * 4, 1, 4, 1, 1, pad_type = args.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        block1 = self.block1(x)                                 # out: batch * 64 * 32 * 32
        block2 = self.block2(block1)                            # out: batch * 128 * 16 * 16
        block3 = self.block3(block2)                            # out: batch * 256 * 8 * 8
        x = self.block4(block3)                                 # out: batch * 512 * 4 * 4
        x = self.final1(x)                                      # out: batch * 512 * 4 * 4
        x = self.final2(x)                                      # out: batch * 1 * 4 * 4
        
        return torch.mean(x)
    
class PatchDiscriminator70_2(nn.Module):
    def __init__(self, args):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(512, args.start_channels, 1, 1, 0, pad_type = args.pad, norm = 'none'),
            Conv2dLayer(args.start_channels, args.start_channels, 7, 1, 3, pad_type = args.pad, norm = args.norm)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(args.start_channels , args.start_channels * 2, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 2, args.start_channels * 2, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(args.start_channels * 2, args.start_channels * 4, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 4, args.start_channels * 4, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(args.start_channels * 4, args.start_channels * 8, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 8, args.start_channels * 8, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(args.start_channels * 8, args.start_channels * 4, 4, 1, 1, pad_type = args.pad, norm = args.norm)
        self.final2 = Conv2dLayer(args.start_channels * 4, 1, 4, 1, 1, pad_type = args.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return torch.mean(x)

class PatchDiscriminator70_3(nn.Module):
    def __init__(self, args):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(320, args.start_channels, 1, 1, 0, pad_type = args.pad, norm = 'none'),
            Conv2dLayer(args.start_channels, args.start_channels, 7, 1, 3, pad_type = args.pad, norm = args.norm)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(args.start_channels , args.start_channels * 2, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 2, args.start_channels * 2, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(args.start_channels * 2, args.start_channels * 4, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 4, args.start_channels * 4, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(args.start_channels * 4, args.start_channels * 8, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 8, args.start_channels * 8, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(args.start_channels * 8, args.start_channels * 4, 4, 1, 1, pad_type = args.pad, norm = args.norm)
        self.final2 = Conv2dLayer(args.start_channels * 4, 1, 4, 1, 1, pad_type = args.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return torch.mean(x)
    
class PatchDiscriminator70_4(nn.Module):
    def __init__(self, args):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(160, args.start_channels, 1, 1, 0, pad_type = args.pad, norm = 'none'),
            Conv2dLayer(args.start_channels, args.start_channels, 7, 1, 3, pad_type = args.pad, norm = args.norm)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(args.start_channels , args.start_channels * 2, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 2, args.start_channels * 2, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(args.start_channels * 2, args.start_channels * 4, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 4, args.start_channels * 4, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(args.start_channels * 4, args.start_channels * 8, 4, 2, 1, pad_type = args.pad, norm = args.norm),
            Conv2dLayer(args.start_channels * 8, args.start_channels * 8, 3, 1, 1, pad_type = args.pad, norm = args.norm)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(args.start_channels * 8, args.start_channels * 4, 4, 1, 1, pad_type = args.pad, norm = args.norm)
        self.final2 = Conv2dLayer(args.start_channels * 4, 1, 4, 1, 1, pad_type = args.pad, norm = 'none', activation = 'none')
        self.classifier = nn.Sequential(nn.Linear(900,100),nn.LeakyReLU(0.2,True),nn.Linear(100,1))
    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        x = self.classifier(x)
        return x


class UresNet(nn.Module):
    def __init__(self, opt):
        super(UresNet, self).__init__()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 1, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 224 * 224
        E2 = self.E2(E1)                                        # out: batch * 128 * 112 * 112
        E3 = self.E3(E2)                                        # out: batch * 256 * 56 * 56
        E4 = self.E4(E3)                                        # out: batch * 512 * 28 * 28
        # Bottle neck
        E4 = self.BottleNeck(E4)                                # out: batch * 512 * 28 * 28
        # Decode the center code
        D1 = self.D1(E4)                                        # out: batch * 256 * 56 * 56
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 56 * 56
        D2 = self.D2(D1)                                        # out: batch * 128 * 112 * 112
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 112 * 112
        D3 = self.D3(D2)                                        # out: batch * 64 * 224 * 224
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 224 * 224
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x


class Unet(nn.Module):
    def __init__(self, args,num_classes):
        super(Unet, self).__init__()

        # The generator is U shaped
        # Encoder
        self.E0 = Conv2dLayer(in_channels = 3,  out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE1 = Conv2dLayer(in_channels = 64,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE11 = Conv2dLayer(in_channels = 128,  out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UE2 = Conv2dLayer(in_channels = 128, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE22 = Conv2dLayer(in_channels = 256, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE3 = Conv2dLayer(in_channels = 256, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE33 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE4 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UE44 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD1 = Conv2dLayer(in_channels = 1024,  out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD11 = Conv2dLayer(in_channels = 256,  out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu', norm = args.norm)
        self.UD2 = Conv2dLayer(in_channels = 512, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD22 = Conv2dLayer(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD3 = Conv2dLayer(in_channels = 256, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD33 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD4 = Conv2dLayer(in_channels = 128, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)
        self.UD44 = Conv2dLayer(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = args.pad, activation = 'relu',norm = args.norm)


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(2)
        self.blockDMT0 = self.make_layer(Block_of_DMT0,1)

        # Decoder
        self.out_conv = Conv2dLayer(in_channels = 64, out_channels = num_classes, kernel_size=1, stride = 1, padding = 0, dilation = 1, pad_type = args.pad, norm = 'none', activation = 'none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #DMT1
        E0 = self.E0(x)
        E0 = self.blockDMT0(E0)
        
        # UE
        UE1 = self.down(E0)
        UE1 = self.UE1(UE1)
        UE1 = self.UE11(UE1)
        UE2 = self.down(UE1)
        UE2 = self.UE2(UE2)
        UE2 = self.UE22(UE2)         
        UE3 = self.down(UE2)
        UE3 = self.UE3(UE3)
        UE3 = self.UE33(UE3)
        UE4 = self.down(UE3)
        UE4 = self.UE4(UE4)
        UE4 = self.UE44(UE4)

        # UD
        UE4 = self.up(UE4)
        UD1=torch.cat((UE4, UE3), 1)
        UD1 = self.UD1(UD1)
        UD1 = self.UD11(UD1)
        UD1 = self.up(UD1)
        UD2 = torch.cat((UD1, UE2), 1)
        UD2 = self.UD2(UD2)
        UD2 = self.UD22(UD2)         
        UD2 = self.up(UD2)
        UD3 = torch.cat((UD2, UE1), 1)
        UD3 = self.UD3(UD3)
        UD3 = self.UD33(UD3)
        UD3 = self.up(UD3)
        UD4 = torch.cat((UD3, E0), 1)
        UD4 = self.UD4(UD4)
        UD4 = self.UD44(UD4)

        x = self.out_conv(UD4)
        return x, UD4,UD4,UD4,UD4,UD4