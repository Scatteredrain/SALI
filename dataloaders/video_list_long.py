import os
from PIL import Image, ImageEnhance
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import time
from glob import glob
import os.path as osp
from mypath import Path
import pdb

def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i]  = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label

def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i]  = imgs[i].rotate(random_angle, mode)
            label[i] = label[i].rotate(random_angle, mode)
    return imgs, label

class VideoFinetuneDataset(data.Dataset):
    def __init__(self, dataset='MoCA', split='TrainDataset', trainsize=256, input_length=10, fsampling_rate=1):
        self.trainsize = trainsize
        self.input_length = input_length
        self.fsampling_rate = fsampling_rate
        self.image_list = []
        self.gt_list = []
        self.pred_list = []
        self.extra_info = []
        self.dataset = dataset

        if dataset == 'SUN-SEG':
            img_root = '/memory/yizhenyu/dataset/SUN/data/SUN-SEG/{}/Frame'.format(split)
            pred_root = '/data/yizhenyu/project/video-polyp-seg/SLT-Net/res/SUN-SEG/{}'.format(split)

            for case in os.listdir(osp.join(img_root)):
                images = sorted(glob(osp.join(img_root,case,'*.jpg')))
                gt_list = sorted(glob(osp.join(img_root.replace('Frame','GT'),case,'*.png')))
                pred_list = sorted(glob(osp.join(pred_root,case+'/Pred','*.png')))
                if pred_list == []:
                    print('no preds:', osp.join(pred_root,case+'/Pred','*.png'))
                frames_to_end = self.fsampling_rate * (self.input_length - 1)
                for i in range(len(images)-frames_to_end):   # 0~len-10 contiguous frames
                    self.extra_info += [ (case, i) ]  # scene and frame_id of first
                    self.gt_list.append(gt_list[i:i + frames_to_end + 1:self.fsampling_rate])
                    self.image_list.append(images[i:i + frames_to_end + 1:self.fsampling_rate])
                    self.pred_list.append(pred_list[i:i + frames_to_end + 1:self.fsampling_rate])

        else:
            if dataset == 'CAD2016':    
                root = Path.db_root_dir('CAD2016')
                img_format = '*.png'
            elif dataset == 'MoCA': 
                root = Path.db_root_dir('MoCA')
                img_format = '*.jpg' #jpg

            # data_root = osp.join(root, 'TrainDataset_per_sq_pvtv2_1e-5_wo_pseudo_epoch_42')
            data_root = osp.join(root, 'MoCA-Video-Train') #MoCA-Video-Train

            for scene in os.listdir(osp.join(data_root)):
                images  = sorted(glob(osp.join(data_root, scene, 'Pred', '*.png')))
                gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))

                frames_to_end = self.fsampling_rate * (self.input_length - 1)

                for i in range(len(images)-frames_to_end):
                    self.extra_info += [ (scene, i) ]  # scene and frame_id
                    self.gt_list.append(gt_list[i:i + frames_to_end + 1:self.fsampling_rate])
                    self.image_list.append(images[i:i + frames_to_end + 1:self.fsampling_rate])
                                                   
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        preds= []
        gts  = []
        names= []
        IMG = None
        PRED= None
        LABEL = None
        index = index % len(self.image_list)

        if self.dataset == 'SUN-SEG':

            for i in range(len(self.pred_list[index])):
                imgs += [self.rgb_loader(self.image_list[index][i])]
                preds+= [self.binary_loader(self.pred_list[index][i])]
                gts  += [self.binary_loader(self.gt_list[index][i])]
                names+= [self.pred_list[index][i].split('/')[-1]]

        else:
            for i in range(len(self.image_list[index])):
                rgb_name = self.image_list[index][i].replace('Pred','Frame')
                rgb_name = rgb_name.replace('.png','.jpg')
                # rgb_name = self.image_list[index][i].replace('Pred','Imgs')
                # rgb_name = rgb_name.replace('.png','.jpg')
                imgs += [self.rgb_loader(rgb_name)]
                preds+= [self.binary_loader(self.image_list[index][i])]
                gts  += [self.binary_loader(self.gt_list[index][i])]
                names+= [self.image_list[index][i].split('/')[-1]]

        scene= self.image_list[index][0].split('/')[-3]  

        # imgs, gts = cv_random_flip(imgs, gts)
        # imgs, gt = randomCrop(imgs, gt)
        # imgs, gts = randomRotation(imgs, gts)

        for i in range(len(imgs)):
            imgs[i] = self.train_transform(imgs[i])
            preds[i] = self.train_transform(preds[i])
            gts[i] = self.train_transform(gts[i])

        for idx, (img, pred, label) in enumerate(zip(imgs, preds, gts)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = pred
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(imgs), *(img.shape))
                PRED = torch.zeros(len(imgs), *(pred.shape))
                LABEL = torch.zeros(len(imgs), *(label.shape))
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = pred
                LABEL[idx, :, :, :] = label
        return IMG, PRED, LABEL

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 

    def __len__(self):
        return len(self.image_list)

# dataloader for training
def get_loader(dataset, split, batchsize, trainsize, input_length, fsampling_rate,
    shuffle=True, num_workers=0, pin_memory=True):
    dataset = VideoFinetuneDataset(dataset, split,trainsize, input_length, fsampling_rate)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, dataset='SUN-SEG', split='TestHardDataset/Seen',
                input_length=10, fsampling_rate=1):
        self.input_length = input_length
        self.fsampling_rate = fsampling_rate
        self.image_list = []
        self.extra_info = []
        self.dataset =dataset
        self.img_root = '/memory/yizhenyu/dataset/SUN/data/SUN-SEG/{}/Frame'.format(split)
        self.pred_root = '/data/yizhenyu/project/video-polyp-seg/SLT-Net/res/SUN-SEG/{}'.format(split)

        if dataset == 'SUN-SEG':
            img_root = '/memory/yizhenyu/dataset/SUN/data/SUN-SEG/TestHardDataset/Seen/Frame'
            for case in os.listdir(osp.join(img_root)):
                images = sorted(glob(osp.join(img_root,case,'*.jpg')))
                clip_size = self.input_length
                skip_size = self.input_length

                video_len = len(images)
                for i in range(0, video_len, skip_size):    
                    clip_im = []
                    indices = list(range(i, min(i+clip_size*(self.fsampling_rate), video_len), self.fsampling_rate))
                    if len(indices) < clip_size:
                        continue
                    for j in indices:
                        clip_im.append(images[j])
                    self.image_list.append(clip_im)

                if video_len % clip_size != 0:
                    clip_im = []
                    for j in list(range(video_len-10,video_len,self.fsampling_rate)):
                        clip_im.append(images[j])
                    self.image_list.append(clip_im)



        else:
            if dataset == 'CAD2016':    
                root = Path.db_root_dir('CAD2016')
                img_format = '*.png'
            elif dataset == 'MoCA': 
                root = Path.db_root_dir('MoCA')
                img_format = '*.jpg'

            data_root = osp.join(root, split)

            for scene in os.listdir(osp.join(data_root)):
                images  = sorted(glob(osp.join(data_root, scene, 'Pred', '*.png')))

                clip_size = self.input_length
                skip_size = self.input_length
                out = False

                video_len = len(images)    # skip the first frame and the last frames
                for i in range(1, video_len, skip_size):
                    clip_im = []
                    indices = list(range(i, min(i+clip_size*(self.fsampling_rate), video_len), self.fsampling_rate))
                    if len(indices) < clip_size:
                        continue
                    for j in indices:
                        clip_im.append(images[j])
                    self.image_list.append(clip_im)

        if len(self.image_list) == 0:
            raise
        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor()])

        self.index = 0
        self.size = len(self.image_list)

    def load_data(self):
        imgs = []
        shts = []
        names= []
        gts = []
        IMG = None
        PRED= None
        LABEL = None
        GT = None
        # forward

        if self.dataset == 'SUN-SEG':
            for i in range(len(self.image_list[self.index])):
                binary_name = self.image_list[self.index][i].replace('Frame','GT').replace('.jpg','.png')
                sht_name = self.pred_root + '/' + binary_name.split('/')[-2] + '/Pred/' + binary_name.split('/')[-1]
                imgs += [self.rgb_loader(self.image_list[self.index][i])]
                gts += [self.binary_loader(binary_name)]
                shts += [self.binary_loader(sht_name)]
                names+= [self.image_list[self.index][i].split('/')[-1]]

        else:
            for i in range(len(self.image_list[self.index])):
                # rgb_name = self.image_list[self.index][i].replace('Pred','Frame')
                rgb_name = self.image_list[self.index][i].replace('Pred','Imgs')
                rgb_name = rgb_name.replace('.png','.jpg')
                imgs += [self.rgb_loader(rgb_name)]
                shts += [self.binary_loader(self.image_list[self.index][i])]
                names+= [self.image_list[self.index][i].split('/')[-1]]

        img_size = imgs[0].size

        for i in range(len(imgs)):
            imgs[i] = self.transform(imgs[i]).unsqueeze(0)
            shts[i] = self.transform(shts[i]).unsqueeze(0)
            gts[i] = self.transform(gts[i]).unsqueeze(0)

        scene= self.image_list[self.index][0].split('/')[-3] 
        self.index += 1
        self.index = self.index % self.size

        for idx, (img, sht, gt) in enumerate(zip(imgs, shts, gts)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = sht
                GT[idx, :, :, :] = gt

            else:
                IMG = torch.zeros(len(imgs), *(img.shape))
                PRED = torch.zeros(len(imgs), *(sht.shape))
                GT = torch.zeros(len(imgs), *(gt.shape))
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = sht
                GT[idx, :, :, :] = gt
        return IMG, PRED, GT, names, scene #img_size, names, scene

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size