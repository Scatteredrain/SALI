from __future__ import print_function, division
import os
import logging
import sys
from tqdm import tqdm
import argparse

sys.path.append('dataloaders')

import torch
import torch.nn.functional as F
import numpy as np
# import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from lib import VideoModel as Network
from utils.utils import clip_gradient, adjust_lr
from dataloaders import video_dataloader,get_loader
from tensorboardX import SummaryWriter


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def cofficent_calculate(preds,gts,threshold=0.5):
    eps = 1e-5
    preds = preds > threshold
    intersection = (preds * gts).sum()
    union =(preds + gts).sum()
    dice = 2 * intersection  / (union + eps)
    iou = intersection/(union - intersection + eps)
    return (dice, iou)

def freeze_network(model):
    for name, p in model.named_parameters():
        if "fusion_conv" not in name:
            p.requires_grad = False


#######################################################################################################################

def train(train_loader, model, optimizer, epoch, save_path, writer, freq):
    """
    train function
    """
    global step
    
    model.train()
    
    loss_all = 0
    epoch_step = 0
    
    try:
        for i, data_blob in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = [x.cuda() for x in data_blob[0]] # Frames*3
            gt_mem = data_blob[1][0].cuda()   
            gt = data_blob[1][1].cuda()   
            case_idx = data_blob[2]

            ## memory 
            # if 0 in case_idx:
            #     model.mem_bank.clear_memory()
            #     model.first_case_gt = gts[0].clone().unsqueeze(0)
            # print('case: ',case_idx)
            # print('mem_t: ',model.mem_bank.T)
            preds = model(images, case_idx, mode='train', gt_mem=gt_mem, use_mem=True, add_mem=False)
            loss = structure_loss(preds[0], gt) + structure_loss(preds[1], gt) + structure_loss(preds[2], gt) + structure_loss(preds[3], gt) 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
        

            if i % 200 == 0 or i == total_step or i == 1:
                print(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                     format(epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                     format(epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0][0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train_RGB', grid_image, step)
                grid_image = make_grid(gt[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train_GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('train_Pred_final', torch.tensor(res), step, dataformats='HW')
       
        
        loss_all /= epoch_step
        print('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        #if epoch % 50 == 0:
        # torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, mem_freq):
    """
    validation function
    """
    global best_mae, best_epoch, best_dice, step_val

    model.eval()
    # model.mem_bank.clear_memory()
    
    with torch.no_grad():
        meandice_case = {}
        mae_sum = 0
        dice_sum = 0
        for i in tqdm(range(test_loader.size)):
            images, gt, name, scene, case_idx = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            images = [x.cuda() for x in images]

            ## memory
            mem_batch_freq = mem_freq  # batchsize=1

            # tqdm.write('case:{}'.format(case_idx))
            # tqdm.write('mem_t:{}'.format(model.module.mem_bank.T))

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
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            dice = cofficent_calculate(res,gt)[0]
            if meandice_case.get(scene) is None:
                    meandice_case[scene] = [] 
            meandice_case[scene].append(dice)
            dice_sum += dice
            # tqdm.write(str(dice))
            step_val += 1
            if i%200 == 0:
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('val_RGB', grid_image, step_val)

                writer.add_image('val_GT', torch.tensor(gt), step_val, dataformats='HW')
                writer.add_image('val_Pred_final', torch.tensor(res), step_val, dataformats='HW')
               

        mae = mae_sum / test_loader.size
        dice = dice_sum / test_loader.size
        Meandice_onCase = [np.mean(meandice_case[scene]) for scene in meandice_case]
        Meandice_onCase = sum(Meandice_onCase)/len(Meandice_onCase)

        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalar('DICE', torch.tensor(dice), global_step=epoch)
        writer.add_scalar('meanDICE_on_cases', torch.tensor(Meandice_onCase), global_step=epoch)
        # print('Epoch: {}, DICE:{}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, dice, mae, best_mae, best_epoch))

        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}_best.pth'.format(epoch))
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} DICE:{} DiceOnCases:{}, MAE:{} bestEpoch:{} bestdice:{}'.format(epoch, dice,Meandice_onCase, mae, best_epoch, best_dice))
        print(
            '[Val Info]:Epoch:{} DICE:{} DiceOnCases:{}, MAE:{} bestEpoch:{} bestdice:{}'.format(epoch, dice,Meandice_onCase, mae, best_epoch, best_dice))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataset_root',  type=str, default='./datasets/SUN-SEG') 
    parser.add_argument('--trainsplit', type=str, default='TrainDataset', help='train dataset')
    parser.add_argument('--testsplit', type=str, default='TestHardDataset/Seen', help='val dataset')
    
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')

    parser.add_argument('--pretrained_weights', type=str, default='./pretrained/pvt_v2_b5.pth',
                        help='path to the pretrained model')
    
    parser.add_argument('--resume', type=str, default='', help='train from checkpoints')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
    
    parser.add_argument('--mem_freq', type=int, default=5, help='mem every n frames')
    parser.add_argument('--test_mem_length', type=int, default=50, help='max num of memory frames')
    
    parser.add_argument('--save_path', type=str,default='./snapshot/SALI/',
                        help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    
    opt = parser.parse_args()

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(filename=os.path.join(save_path,'train.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    # build the model
    model = Network(opt).cuda()
    model = torch.nn.DataParallel(model)
    logging.info('more than one gpu!')

    # pdb.set_trace()
    logging.info('save to {}'.format(save_path))
    logging.info("Network-Train")

    #freeze_network(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)


    # load data
    logging.info('===> Loading datasets')
    print('===> Loading datasets')

    train_loader, val_loader = video_dataloader(opt)
    total_step = len(train_loader)

    # logging
    
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.resume, save_path, opt.decay_epoch))
    

    step = 0
    step_val = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_dice = 0
    best_epoch = 0

    freq = opt.mem_freq // opt.batchsize * opt.batchsize
    skip_list = [10,15,20,25,5]
    skip_i = 0


    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        if not opt.valonly:
            train(train_loader, model, optimizer, epoch, save_path, writer, freq)
        if isinstance(model,torch.nn.DataParallel):
            model = model.module
        model.cuda(0)
        val(val_loader, model, epoch, save_path, writer, opt.mem_freq)
        model.mem_bank.clear_memory()
        model = torch.nn.DataParallel(model,device_ids=[0,1]) 

        if epoch % 5 == 0:
            train_loader = get_loader(dataset=opt.dataset,
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              train_split=opt.trainsplit,
                              max_skip=skip_list[skip_i])
            
            logging.info('===> Reloading TrainDatasets of max_skip:{}'.format(skip_list[skip_i]))
            print('===> Reloading TrainDatasets of max_skip:{}'.format(skip_list[skip_i]))
            skip_i = min(4,skip_i+1)
