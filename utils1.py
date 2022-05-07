import torch
from torch.nn import init
import network

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		size = m.weight.size()
		m.weight.data.normal_(0.0, 0.1)
		m.bias.data.fill_(0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant(m.bias.data, 0.0)

def lr_scheduler(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer

def exp_lr_scheduler(optimizer, epoch, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
    
def create_generator(args,nclass):
    # Initialize the network
    generator = network.Generator(args,nclass)
    # Init the network
    network.weights_init(generator, init_type = args.init_type, init_gain = args.init_gain)
    print('Generator is created!')
    return generator



def create_discriminator(args):
    # Initialize the network
    discriminator = network.VGG128_Discriminator(in_nc = 160,base_nf = 64)
    # Init the network
    network.weights_init(discriminator, init_type = args.init_type, init_gain = args.init_gain)
    print('Discriminators is created!')
    return discriminator


def create_discriminator1(args):
    # Initialize the network
    discriminator = network.PatchDiscriminator70_1(args)
    # Init the network
    network.weights_init(discriminator, init_type = args.init_type, init_gain = args.init_gain)
    print('Discriminators1 is created!')
    return discriminator

def create_discriminator2(args):
    # Initialize the network
    discriminator = network.PatchDiscriminator70_2(args)
    # Init the network
    network.weights_init(discriminator, init_type = args.init_type, init_gain = args.init_gain)
    print('Discriminators2 is created!')
    return discriminator


def create_discriminator3(args):
    # Initialize the network
    discriminator = network.PatchDiscriminator70_3(args)
    # Init the network
    network.weights_init(discriminator, init_type = args.init_type, init_gain = args.init_gain)
    print('Discriminators3 is created!')
    return discriminator

def create_discriminator4(args):
    # Initialize the network
    discriminator = network.PatchDiscriminator70_4(args)
    # Init the network
    network.weights_init(discriminator, init_type = args.init_type, init_gain = args.init_gain)
    print('Discriminators4 is created!')
    return discriminator

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net