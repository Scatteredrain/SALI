
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

# cur_file_path=os.path.realpath(__file__)
# cur_directory=os.path.dirname(cur_file_path)
# sys.path.append(os.path.join(cur_directory, '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',  type=str, default='/memory/yizhenyu/dataset/SUN/data/SUN-SEG')
parser.add_argument('--testsplit',  type=str, default='TestHardDataset/Seen')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='training size')
parser.add_argument('--pth_path', type=str, default='./snapshot/LSINet.pth')
parser.add_argument('--pretrained_image_model', default=None,
                        help='path to the pretrained image model')
parser.add_argument('--pvtv2_pretrained_path', default=None,
                        help='whether load pvt_v2_b5')
parser.add_argument('--mem_freq', type=int, default=5, help='mem every n frames')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

if __name__ == '__main__':
    # pthList = ['./snapshot/cash_new_new_b14/Net_epoch_23.pth','./snapshot/cash_new_new_b14/Net_epoch_24.pth',
    #     './snapshot/cash_new_new_b14/Net_epoch_26.pth','./snapshot/cash_new_new_b14/Net_epoch_27.pth'
    #            ,'./snapshot/cash_new_new_b14/Net_epoch_28.pth','./snapshot/cash_new_new_b14/Net_epoch_29.pth'
    #            ,'./snapshot/cash_new_new_b14/Net_epoch_30.pth']
    pthList = [opt.pth_path]
    test_loader = test_dataloader(opt)
    for pth_path in pthList:
        save_root = './res/eval2/{}/{}/{}/'.format(pth_path.split('/')[-1][:-4],opt.dataset_root.split('/')[-1],opt.testsplit)
        # pdb.set_trace()
        os.makedirs(save_root,exist_ok=True)
        save_path = './'+ save_root.split('/')[1] +'/' +save_root.split('/')[2] + '/' + save_root.split('/')[3]
        model = Network(opt)
        # model = torch.nn.DataParallel(model)
        print('loading from:{}'.format(pth_path))
        model.load_state_dict(torch.load(pth_path,map_location='cuda:0'))
        # pretrained_dict = torch.load(opt.pth_path)
        # model_dict = model.state_dict()
        # #pdb.set_trace()
        # for k, v in pretrained_dict.items():
        #     pdb.set_trace()
        if isinstance(model,torch.nn.DataParallel):
            model = model.module
        model.eval()
        model.cuda(0)                  

        ## memory
        mem_batch_freq = opt.mem_freq  # batchsize=1
        # compute parameters
        print('Total Params = %.2fMB' % count_parameters_in_MB(model))

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
                # elif (case_idx % mem_batch_freq == 0) and case_idx!=0:
                #     res= model(images,[case_idx],mode='val', use_mem=True, add_mem=True)[-1]
                # else:
                #     res = model(images,[case_idx],mode='val', use_mem=True, add_mem=False)[-1]
                
                # tqdm.write('T:{}'.format(str(model.mem_bank.T)))
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                dice = cofficent_calculate(res,gt)[0]
                dice_sum += dice
                if dice<0.7:
                    name =name[0].replace('jpg','png')
                    low_res.append([os.path.join(scene,name),dice])
                    #save
                    # save_path_final = save_path + 'Pred/' + scene
                    # if not os.path.exists(save_path_final):
                    #     os.makedirs(save_path_final,exist_ok=True)
                    # imageio.imwrite(os.path.join(save_path_final,name), np.uint8(res*255))
                    tqdm.write(scene+'/'+name)
                    tqdm.write(str(dice))
                
                if reses.get(scene) is None:
                    reses[scene] = [] 
                reses[scene].append(dice)

                ## save 
                # if len(name) == 2:
                #     name = name[0].replace('jpg','png')
                # else: 
                #     name = name.replace('jpg','png')

                # save_path_final = os.path.join(save_root,scene )
                # if not os.path.exists(save_path_final):
                #     os.makedirs(save_path_final,exist_ok=True)
                # # tqdm.write(os.path.join(save_path_final,name))
                # imageio.imwrite(os.path.join(save_path_final,name), np.uint8(res*255))

        meandice = dice_sum/test_loader.size
        print('meandice on frames:{:.4f}'.format(meandice))
        Meandice = 0.0
        dataset,split = opt.testsplit.split('/')
        with open(os.path.join(save_path,'{}_{}_results.txt'.format(dataset,split)), 'w+') as file_to_write:
            file_to_write.write('\n'+'save to :{}'.format(save_root)+'\n')
            file_to_write.write('\n'+'mem_every:{}'.format(str(mem_batch_freq))+'\n')
            for fold_name in reses:
                lens = len(reses[fold_name])
                meandice_1 = np.mean(reses[fold_name][:lens//3])
                meandice_2 = np.mean(reses[fold_name][lens//3:2*lens//3])
                meandice_3 = np.mean(reses[fold_name][2*lens//3:])
                # print(str(fold_name)+'__1/3:{:.3f}'.format(meandice_1))
                # print(str(fold_name)+'__2/3:{:.3f}'.format(meandice_2))
                # print(str(fold_name)+'__3/3:{:.3f}'.format(meandice_3))
                file_to_write.write(str(fold_name)+'\n')
                file_to_write.write(str(fold_name)+'__1/3:{:.3f}'.format(meandice_1)+'\n')
                file_to_write.write(str(fold_name)+'__2/3:{:.3f}'.format(meandice_2)+'\n')
                file_to_write.write(str(fold_name)+'__3/3:{:.3f}'.format(meandice_3)+'\n')
                Meandice += np.mean(reses[fold_name])
                
            Meandice = Meandice / len(reses)
            print('Overall_Meandice on cases:{:.4f}'.format(Meandice))
            file_to_write.write('\n'+'Overall_Meandice on cases:{:.4f}'.format(Meandice)+'\n')
            
            file_to_write.write('\n'+'meandice on frames :{:.4f}'.format(meandice)+'\n')
            for i in low_res:
                file_to_write.write('\n'+'dice:{:.4f}, path:{}'.format(i[1],i[0])+'\n')
        


        # if case_idx != 0:

        #     res_neibor0 = F.upsample(res_neibor[-1], size=gt.shape, mode='bilinear', align_corners=False)
        #     res_neibor0 = res_neibor0.sigmoid().data.cpu().numpy().squeeze()
        #     res_neibor0 = (res_neibor0 - res_neibor0.min()) / (res_neibor0.max() - res_neibor0.min() + 1e-8)
            
        #     name =name[0].replace('jpg','png')
        #     save_path_neibor = save_root + scene + '/Pred_neibor'
        #     # if name[-5] in ['0','5']:
        #     imageio.imwrite(os.path.join(save_path_neibor,name), np.uint8(res*255))
            
        #     res_mem0 = F.upsample(res_mem[-1], size=gt.shape, mode='bilinear', align_corners=False)
        #     res_mem0 = res_mem0.sigmoid().data.cpu().numpy().squeeze()
        #     res_mem0 = (res_mem0 - res_mem0.min()) / (res_mem0.max() - res_mem0.min() + 1e-8)
            
        #     name =name[0].replace('jpg','png')
        #     save_path_mem = save_root + scene + '/Pred_mem'
        #     # if name[-5] in ['0','5']:
        #     imageio.imwrite(os.path.join(save_path_mem,name), np.uint8(res*255))


