import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from lib import VideoModel as Network
from dataloaders import test_dataloader
import imageio
import pdb
from tqdm import tqdm

import sys
sys.path.append('dataloaders')

def cofficent_calculate(pred,gts,threshold=0.5):
    eps = 1e-5
    preds = pred > threshold
    intersection = (preds * gts).sum()
    union =(preds + gts).sum()
    dice = 2 * intersection  / (union + eps)
    iou = intersection/(union - intersection + eps)
    return (dice, iou)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',  type=str, default='/memory/yizhenyu/dataset/SUN/data/SUN-SEG')
parser.add_argument('--testsplit',  type=str, default='TestHardDataset/Seen')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='training size')
parser.add_argument('--pth_path', type=str, default='./snapshot/Net_epoch_8_best.pth')
parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to the pretrained model')
parser.add_argument('--mem_freq', type=int, default=5, help='mem every n frames')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

if __name__ == '__main__':
    
    test_loader = test_dataloader(opt)
    pth_path = opt.pth_path
    save_root = './res/{}/{}/{}'.format(pth_path.split('/')[-1][:-4],opt.dataset_root.split('/')[-1],opt.testsplit)
    # pdb.set_trace()
    os.makedirs(save_root,exist_ok=True)
    model = Network(opt)
    # model = torch.nn.DataParallel(model)
    print('loading from:{}'.format(pth_path))
    model.load_state_dict(torch.load(pth_path,map_location='cuda:0'))

    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model.eval()
    model.cuda(0)                  

    ## memory
    mem_batch_freq = opt.mem_freq  # batchsize=1
    
    reses = {}
    low_res = []
    dice_sum = 0.0
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            images, gt, name, scene, case_idx = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            images = [x.cuda() for x in images]
            
            # print(model.mem_bank.T)
            
            if case_idx == 0:
                model.mem_bank.clear_memory()
                res = model(images,[case_idx],mode='val', use_mem=False, add_mem=True)[-1]
            elif (case_idx % mem_batch_freq == 0) and case_idx!=0:
                res= model(images,[case_idx],mode='val', use_mem=True, add_mem=True)[-1]
            else:   
                res= model(images,[case_idx],mode='val', use_mem=True, add_mem=False)[-1]

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            dice = cofficent_calculate(res,gt)[0]
            dice_sum += dice

            name =name[0].replace('jpg','png')
            save_path_final = save_root + '/' + scene
            if not os.path.exists(save_path_final):
                os.makedirs(save_path_final,exist_ok=True)
            imageio.imwrite(os.path.join(save_path_final,name), np.uint8(res*255))

    meandice = dice_sum/test_loader.size
    print('meandice on frames:{:.4f}'.format(meandice))
