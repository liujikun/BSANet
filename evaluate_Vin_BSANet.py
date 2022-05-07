import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import cv2
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from model.deeplab_multi_ISPRS import DeeplabMulti_ISPRS
from dataset.Vin_dataset import VinDataSet_Test,PotDataSet_Test
from collections import OrderedDict
import os
from PIL import Image
import utils1
from model.discriminator import FCDiscriminator
import torch.nn as nn
import scipy.io
import tifffile
IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)

# DATA_DIRECTORY = './data/Vin'
# DATA_LIST_PATH = './dataset/Vin_list/train.txt'
# SAVE_PATH = './result/Vin'

DATA_DIRECTORY = './data/Vin'
DATA_LIST_PATH = './dataset/Vin_list/cut.txt'
SAVE_PATH = './result/Vin'

# DATA_DIRECTORY = './data/Pot'
# DATA_LIST_PATH = './dataset/Pot_list/train.txt'
# SAVE_PATH = './result/Pot'

# DATA_DIRECTORY = './data/Pot'
# DATA_LIST_PATH = './dataset/Pot_list/cut.txt'
# SAVE_PATH = './result/Pot'

IGNORE_LABEL = 255
NUM_CLASSES = 5
NUM_STEPS = 36 # Number of images in the validation set.




palette = [188, 188, 188, 215,252, 0, 0, 255, 0, 255, 0, 0, 139, 0, 139]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def mIou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]])#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    # print(inputTmp.shape)
    # print(input.shape)
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    target[target>4]=4
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            # print(union)
            iou = intersection / union
            ious.append(iou)
        # print(ious)
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return miou


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default='./Vin2Pot_our.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    parser.add_argument('--dila', type = int, default = 2, help = '2 for colorization, 3 for other tasks')
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    
    args = get_arguments()
    device = torch.device("cuda" if not args.cpu else "cpu")
    if not os.path.exists(args.save):
        os.makedirs(args.save)


    model = utils1.create_generator(args,args.num_classes)


    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    
    model = model.to(device)

    model.eval()
    if 'Pot' in SAVE_PATH:
        testloader = data.DataLoader(PotDataSet_Test(args.data_dir, args.data_list, crop_size=512, mean=IMG_MEAN, scale=False, mirror=False),
                                        batch_size=1, shuffle=False, pin_memory=True)
    else:
        testloader = data.DataLoader(VinDataSet_Test(args.data_dir, args.data_list, crop_size=512, mean=IMG_MEAN, scale=False, mirror=False),
                                        batch_size=1, shuffle=False, pin_memory=True)                                    
    size_height=[2569,2557,2566,2575,2558,2565,2565,2565,1281,2315,2546,2546,2546,2546,1783,3007,1995,2550,2557,2557,2557,2557,2557]
    size_width=[1919,1887,1893,1922,2818,1919,1919,1919,2336,1866,1903,1903,1903,1903,2995,2006,1996,3816,1887,1887,1887,1887,1887]
    # size_width=[2428,1917,1917,1917,1934,1980,1980,1581,1388,2805]
    # size_height=[2767,3313,2567,2563,2563,2555,2555,2555,2555,1884]
    miou = 0
    num = 0
    for index, batch in enumerate(testloader):
        num = num + 1
        if index % 10 == 0:
            print('%d processd' % index)
        image, label, _,name = batch
        # print(torch.Tensor.size(image))
        image = image.to(device)
        # interp = nn.Upsample(size=(size_height[index], size_width[index]), mode='bilinear', align_corners=True)
        interp = nn.Upsample(size=(1024,1024), mode='bilinear', align_corners=True)


        output,D4_out,_,merge,_,_= model(image)
        pred = torch.argmax(F.softmax(output), dim=1, keepdim=False).cpu()
        label = label.long().cpu()
        Iou = mIou(pred,label,NUM_CLASSES)
        miou = miou + Iou
        output = interp(F.softmax(output)).cpu().data[0].numpy()


        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)


        image = image.cpu().data[0].numpy().astype(np.float32)
        image = image * 255.0
        image = image.transpose(1,2,0).astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        label = label.data[0].numpy().astype(np.uint8) 
        gt = colorize_mask(label)
        

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s_try_color.png' % (args.save, name.split('.')[0]))
        image.save('%s/%s_src.png' % (args.save, name.split('.')[0]))
        gt.save('%s/%s_gt.png' % (args.save, name.split('.')[0]))
    print('Miou:',miou/num)

if __name__ == '__main__':
    main()
