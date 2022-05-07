import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2

class VinDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=512, mean=(0.485, 0.456, 0.406), scale=True, mirror=True, ignore_label=8):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4,8:255}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

     # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, image, label):

        # rotate
        rotate = random.randint(0, 3)
        if rotate != 0:
            image = np.rot90(image, rotate)
            label = np.rot90(label, rotate)
        # horizontal flip
        if np.random.random() >= 0.5:
            image = cv2.flip(image, flipCode = 1)
            label = cv2.flip(label, flipCode = 1)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)


        # crop and aug
   
        h, w = image.shape[:2]    
        min_len = min(h,w)
        min_len = min(1024,min_len)
        rand_h, rand_w = self.random_crop_start(h, w, min_len, 4)
        # rand_h, rand_w = 0,0
        image = image[rand_h:rand_h+min_len, rand_w:rand_w+min_len]
        label = label[rand_h:rand_h+min_len, rand_w:rand_w+min_len]

        image = cv2.resize(image, (self.crop_size,self.crop_size), interpolation=cv2.INTER_CUBIC) 
        label = cv2.resize(label, (self.crop_size,self.crop_size), interpolation=cv2.INTER_NEAREST)  
        image, label = self.img_aug(image, label)




        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0 
        # image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


class VinDataSet_Test(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=512, mean=(0.485, 0.456, 0.406), scale=True, mirror=True, ignore_label=8):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4,8:255}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "cut_images/%s" % name)
            label_file = osp.join(self.root, "cut_labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

     # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, image, label):

        # rotate
        rotate = random.randint(0, 3)
        if rotate != 0:
            image = np.rot90(image, rotate)
            label = np.rot90(label, rotate)
        # horizontal flip
        if np.random.random() >= 0.5:
            image = cv2.flip(image, flipCode = 1)
            label = cv2.flip(label, flipCode = 1)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # crop and aug
        h, w = image.shape[:2]    
        min_len = min(h,w)
        min_len = min(1024,min_len)
        rand_h, rand_w = self.random_crop_start(h, w, min_len, 4)
        rand_h, rand_w = 0,0
        image = image[rand_h:rand_h+min_len, rand_w:rand_w+min_len]
        label = label[rand_h:rand_h+min_len, rand_w:rand_w+min_len]

        image = cv2.resize(image, (self.crop_size,self.crop_size), interpolation=cv2.INTER_CUBIC) 
        label = cv2.resize(label, (self.crop_size,self.crop_size), interpolation=cv2.INTER_NEAREST)  
        # image, label = self.img_aug(image, label)




        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0
        # print(image.shape)
        # mean_image = (image[:,:,0].mean(), image[:,:,1].mean(),image[:,:,2].mean()) 
        # print(mean_image)
        # image -= mean_image
        # image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


class PotDataSet_Test(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=512, mean=(0.485, 0.456, 0.406), scale=True, mirror=True, ignore_label=8):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4,8:255}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

     # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, image, label):

        # rotate
        rotate = random.randint(0, 3)
        if rotate != 0:
            image = np.rot90(image, rotate)
            label = np.rot90(label, rotate)
        # horizontal flip
        if np.random.random() >= 0.5:
            image = cv2.flip(image, flipCode = 1)
            label = cv2.flip(label, flipCode = 1)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # crop and aug
        h, w = image.shape[:2]
        rand_h, rand_w = self.random_crop_start(h, w, self.crop_size * 4, 4)
        rand_h, rand_w = 0,0
        image = image[rand_h:rand_h+4*self.crop_size, rand_w:rand_w+4*self.crop_size]
        label = label[rand_h:rand_h+4*self.crop_size, rand_w:rand_w+4*self.crop_size]

        image = cv2.resize(image, (self.crop_size,self.crop_size), interpolation=cv2.INTER_CUBIC) 
        label = cv2.resize(label, (self.crop_size,self.crop_size), interpolation=cv2.INTER_NEAREST)  
        # image, label = self.img_aug(image, label)




        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0
        # print(image.shape)
        # mean_image = (image[:,:,0].mean(), image[:,:,1].mean(),image[:,:,2].mean()) 
        # print(mean_image)
        # image -= mean_image
        # image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name