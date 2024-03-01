import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import time
from glob import glob
import os.path as osp
import pdb


# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(label)):
            label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)
       
    return imgs, label

def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    for i in range(len(label)):
        label[i] = label[i].crop(random_region)

    return imgs, label

def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)

        for i in range(len(label)):
            label[i] = label[i].rotate(random_angle, mode)

    return imgs, label

def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

class VideoDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize=352, batchsize=8, split='MoCA-Video-Train',max_skip=5):
        self.trainsize = trainsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []
        self.case_idx = []
        

        img_root = '{}/{}/Frame'.format(dataset_root, split)
        for case in os.listdir(osp.join(img_root)):
            images = sorted(glob(osp.join(img_root,case,'*.jpg')))
            gt_list = sorted(glob(osp.join(img_root.replace('Frame','GT'),case,'*.png')))

            
            # for the first frame of this case
            ##########
            i=0
            self.extra_info += [ (case, i) ]  # scene and frame_id
            self.gt_list    += [ [gt_list[i+1],
                                gt_list[i]] ] 
            self.image_list += [ [images[i+1],
                                images[i], 
                                images[i+1]] ]
            self.case_idx += [0]
            #########


            for i in range(1,len(images)-1):
                
                mem_idx = random.randint(max(i-max_skip,0),i-1)

                self.extra_info += [ (case, i) ]  # scene and frame_id
                self.gt_list    += [ [gt_list[mem_idx],
                                    gt_list[i] ]]
                self.image_list += [ [images[mem_idx],
                                        images[i], 
                                        images[i+1]] ]
                self.case_idx += [i]

            # for the last frame of this case
            ##########
            i = len(images)-1
            mem_idx = random.randint(max(i-1-max_skip,0),i-2)
            self.gt_list    += [ [gt_list[mem_idx],
                                    gt_list[i]] ]
            self.image_list += [ [images[mem_idx],
                                    images[i], 
                                    images[i-1]] ]
            self.case_idx += [i]
            ################

            # if len(images) % batchsize != 0:
            #     for i in range(1,batchsize - len(images)%batchsize + 1): # skip the first frame
            #         self.extra_info += [ (case, i) ]  # scene and frame_id
            #         self.gt_list    += [ gt_list[i] ]
            #         self.image_list += [ [images[i], 
            #                        images[i+1]] ]
            #         self.case_idx += [i]

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        names= []
        gts = []
        index = index % len(self.image_list)

        for i in range(len(self.image_list[index])):
            imgs += [self.rgb_loader(self.image_list[index][i])]
            names+= [self.image_list[index][i].split('/')[-1]]
        for i in range(len(self.gt_list[index])):
            gts +=[self.binary_loader(self.gt_list[index][i])]
        #print(names)
        scene= self.image_list[index][0].split('/')[-3]  


        imgs, gts = cv_random_flip(imgs, gts)
        #imgs, gt = randomCrop(imgs, gt)
        imgs, gts = randomRotation(imgs, gts)
        imgs = colorEnhance(imgs)

        for i in range(len(imgs)):
            imgs[i] = self.img_transform(imgs[i])
        for i in range(len(gts)):
            gts[i] = randomPeper(gts[i])
            gts[i] = self.gt_transform(gts[i])

        case_idx = self.case_idx[index]

        return imgs, gts, case_idx#, scene, names

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
def get_loader(dataset_root, batchsize, trainsize, train_split,
    shuffle=True, num_workers=12, pin_memory=True, max_skip=5):

    dataset = VideoDataset(dataset_root, trainsize,batchsize,split=train_split,max_skip=max_skip)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, dataset_root, split='TestHardDataset/Seen', testsize=352):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []
        self.case_idx = [] 

        
        img_root = dataset_root + '/{}/Frame'.format(split)
        for case in os.listdir(osp.join(img_root)):
            images = sorted(glob(osp.join(img_root,case,'*.jpg')))
            gt_list = sorted(glob(osp.join(img_root.replace('Frame','GT'),case,'*.png')))
            
            for i in range(len(images)-1):
                # self.extra_info += [ (case, i) ]  # scene and frame_id
                self.gt_list    += [ gt_list[i] ]
                self.image_list += [ [images[i], 
                                    images[i+1]] ]
                self.case_idx += [i]
                # if 'case_M_20181010094822_0U62363101085921_1_003_001-1_a10_ayy_image0096' in images[i]:
                #     print('!!!!!!!!')

            
            for i in range(len(images)-1, len(images)-2,-1): 
                self.gt_list    += [ gt_list[i] ]
                self.image_list += [ [images[i], 
                                images[i-1]] ]
                self.case_idx += [i]


        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        names= []

        # if 'case_M_20181010094822_0U62363101085921_1_003_001-1_a10_ayy_image0096' in self.image_list[self.index]:
        #     print('!!!!!!!!')
        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]           
            names+= [self.image_list[self.index][i].split('/')[-1]]
            imgs[i] = self.transform(imgs[i]).unsqueeze(0)

        # scene= self.image_list[self.index][0].split('/')[-3]  
        scene= self.image_list[self.index][0].split('/')[-2]   # SUN-SEG

        gt = self.binary_loader(self.gt_list[self.index])

        case_idx = self.case_idx[self.index]

        self.index += 1
        self.index = self.index % self.size
        return imgs, gt, names, scene, case_idx

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
        