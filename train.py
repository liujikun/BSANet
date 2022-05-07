import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import utils1
from model.deeplab_multi_ISPRS import DeeplabMulti_ISPRS
from model.discriminator import FCDiscriminator,PixelDiscriminator,Discriminator,layoutDiscriminator
from utils.loss import CrossEntropy2d
from utils1 import load_dict
from dataset.Vin_dataset import VinDataSet
from dataset.Pot_dataset import PotDataSet
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 8
## My sets
DATA_DIRECTORY = './data/Vim'
DATA_LIST_PATH = './dataset/Vin_list/train.txt'

IGNORE_LABEL = 255
INPUT_SIZE = 512
DATA_DIRECTORY_TARGET = './data/Pot'
DATA_LIST_PATH_TARGET = './dataset/Pot_list/train.txt'

INPUT_SIZE_TARGET = 512
## My sets
LEARNING_RATE = 3e-5
MOMENTUM = 0.9
NUM_CLASSES = 5
NUM_STEPS = 15000
NUM_STEPS_STOP = 15000  # early stopping
POWER = 0.9
RANDOM_SEED = 1265
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 0.5*LEARNING_RATE
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.5
LAMBDA_ADV_TARGET2 = 0.5
LAMBDA_ADV_TARGET3 = 0
LAMBDA_ADV_TARGET4 = 0
LAMBDA_ADV_TARGET5 = 0


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=int, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target3", type=float, default=LAMBDA_ADV_TARGET3,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target4", type=float, default=LAMBDA_ADV_TARGET4,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target5", type=float, default=LAMBDA_ADV_TARGET5,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--resume", type=bool, default=False,
                        help="resume training or not.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--dila', type = int, default = 2, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation type of networks')
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    input_size = args.input_size
    input_size_target = args.input_size_target

    cudnn.enabled = True

    # Create network

    model = utils1.create_generator(args,args.num_classes)
    if args.resume:
        checkpoint_path = './snapshots/VIN_6000'
        # Load a pre-trained network
        pretrained_net = torch.load(checkpoint_path + '.pth')
        load_dict(model, pretrained_net)
        print('Generator is loaded!')


    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = layoutDiscriminator(num_classes=args.num_classes).to(device)
    model_D2 = Discriminator(num_classes=args.num_classes).to(device)

    model_D1.train()
    model_D1.to(device)
    model_D2.train()
    model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        PotDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(VinDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)
    # print(len(targetloader))

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = (0.9, 0.999), weight_decay = args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.999), weight_decay = args.weight_decay)
    optimizer_D1.zero_grad()


    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.999), weight_decay = args.weight_decay)
    optimizer_D2.zero_grad()


    WEIGHTS = torch.ones(args.num_classes)
    # original weight
    WEIGHTS[0] = 1/0.2802
    WEIGHTS[1] = 1/0.2142
    WEIGHTS[2] = 1/0.2308
    WEIGHTS[3] = 1/0.2622
    WEIGHTS[4] = 1/0.0126
    # WEIGHTS[0] = 1/0.3150
    # WEIGHTS[1] = 1/0.2290
    # WEIGHTS[2] = 1/0.1656
    # WEIGHTS[3] = 1/0.2716
    # WEIGHTS[4] = 1/0.0189
    # WEIGHTS[4] = 1/0.25
    weights = WEIGHTS.cuda()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(weight=weights,ignore_index=255)


    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        loss_adv_target_value3 = 0
        loss_D_value3 = 0

        loss_adv_target_value4 = 0
        loss_D_value4 = 0

        loss_adv_target_value5 = 0
        loss_D_value5 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G
            for param in model.parameters():
                param.requires_grad = True
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)

            pred1, merge_src, concat_src, final_src, U_middle_src, W_middle_src = model(images)


            loss_seg1 = seg_loss(F.softmax(pred1), labels)
            loss = loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward(retain_graph=True)
            loss_seg_value1 += loss_seg1.item() / args.iter_size


            # train with target

            _, batch = targetloader_iter.__next__()
            images,_, _, _ = batch

            images = images.to(device)

            pred_target1, merge_tar, concat_tar, final_tar, U_middle_tar, W_middle_tar = model(images)

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(final_tar)

            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward(retain_graph=True)
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size

            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

            optimizer.step()
            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            pred1 = pred1.detach()

            D_out1 = model_D1(F.softmax(pred1))
            D_out2 = model_D2(final_src)


            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))


            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2


            loss_D1.backward(retain_graph=True)
            loss_D2.backward(retain_graph=True)

            
            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target1 = pred_target1.detach()

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(final_tar)


            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))


            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2


            loss_D1.backward(retain_graph=True)
            loss_D2.backward()


            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()


            optimizer_D1.step()
            optimizer_D2.step()



        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_adv3 = {3:.3f} loss_D3 = {7:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}  loss_adv4 = {4:.3f}, loss_adv5 = {5:.3f} loss_D4 = {6:.3f} loss_D5 = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_adv_target_value3, loss_D_value3, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2,loss_adv_target_value4, loss_adv_target_value5, loss_D_value4, loss_D_value5))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VIN_'+ args.model + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VIN_' + args.model + str(i_iter) + '.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
