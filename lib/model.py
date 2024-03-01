import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.pvtv2_afterTEM import Network as Encoder
from lib.short_term import SRA
from lib.DCN_Module import DCNv2Pack
from lib.long_term import MemoryBank

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class NonLocalNet(nn.Module):
    def __init__(self, in_planes, bn_layer=True, pyramid_type='conv'):
        super(NonLocalNet, self).__init__()
        self.pyramid_type = pyramid_type
        self.nl_layer1 = SRA(in_planes*2, inter_channels=None,
            sub_sample=True, bn_layer=bn_layer, sub_sample_scale=2)
        self.nl_layer2 = SRA(in_planes*2, inter_channels=None,
            sub_sample=True, bn_layer=bn_layer, sub_sample_scale=4)
        self.nl_layer3 = SRA(in_planes*2, inter_channels=None,
            sub_sample=True, bn_layer=bn_layer, sub_sample_scale=8)

        self.conv1 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv2 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv3 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)

    def forward(self, fea1, fea2):
        #pdb.set_trace()
        out1 = self.conv1(self.nl_layer1(torch.cat([fea1[0], fea2[0]], dim=1)))  # fea1/2[0]: [1,32,H,W] -> out1: [1,32,H,W]
        out2 = self.conv2(self.nl_layer2(torch.cat([fea1[1], fea2[1]], dim=1)))
        out3 = self.conv3(self.nl_layer3(torch.cat([fea1[2], fea2[2]], dim=1)))
        return out1, out2, out3
    
    
class MemBranch(nn.Module):
    def __init__(self, indim=32, num=3):
        super(MemBranch, self).__init__()
        self.num = num
        self.Key0 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key1 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key2 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)

        self.Value0 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value1 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value2 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)
  
 
    def forward(self, x):  

        key = [
            self.Key0(x[0]), self.Key1(x[1]), self.Key2(x[2])
        ]
        value = [
            self.Value0(x[0]), self.Value1(x[1]), self.Value2(x[2])
        ]

        return key, value
    
class LongShort_term_fusion(nn.Module):
    def __init__(self, indim=32):
        super(LongShort_term_fusion, self).__init__()
        self.layer1 = nn.Sequential(
                                nn.Conv2d(2*indim, indim, 3, 1, 1),
                                nn.Conv2d(indim, indim, 1, 1, 0)
                                )
        self.layer2 = nn.Sequential(
                                nn.Conv2d(2*indim, indim, 3, 1, 1),
                                nn.Conv2d(indim, indim, 1, 1, 0)
                                )
        self.layer3 = nn.Sequential(
                                nn.Conv2d(2*indim, indim, 3, 1, 1),
                                nn.Conv2d(indim, indim, 1, 1, 0)
                                )
        
    def forward(self, f1,f2):  
        x1 = self.layer1(torch.cat([f1[0],f2[0]],dim=1))
        x2 = self.layer2(torch.cat([f1[1],f2[1]],dim=1))
        x3 = self.layer3(torch.cat([f1[2],f2[2]],dim=1))

        return x1, x2, x3

class VideoModel(nn.Module):
    def __init__(self, args):
        super(VideoModel, self).__init__()

        self.args = args
        self.extra_channels = 0
        print("Select mask mode: concat, num_mask={}".format(self.extra_channels))

        self.backbone = Encoder(pvtv2_pretrained_path=args.pvtv2_pretrained_path, imgsize=self.args.trainsize)
        if self.args.pretrained_image_model is not None:
            self.load_backbone(self.args.pretrained_image_model)

        self.nlnet = NonLocalNet(in_planes=32, pyramid_type='conv')

        self.pcd_align = PCDAlignment(num_feat=32, deformable_groups=8)

        # self.mem_bank = MemoryBank(test_mem_length=args.test_mem_length,num_values=3)
        self.mem_bank = MemoryBank()
        self.first_case_gt = None

        self.membranch = MemBranch(indim=32)

        # self.ls_fusion_conv = LongShort_term_fusion(indim=32)

        self.freeze_bn()

    def load_backbone(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        print("Load pretrained image model parameters from {}".format(pretrained))
        # for k, v in pretrained_dict.items():
        #     if (k in model_dict):
        #         print("load:%s"%k)
            # else:
            #     print("jump over:%s"%k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()
        


    def forward(self, x, case_idx=0, mode='train' , gt_mem=None,use_mem=False, add_mem=False):
        
        ## train
        if mode == 'train':
            image_mem, image_cur, image_neibor = x[0],x[1],x[2]
            fmap_cur=self.backbone.feat_net(image_cur)
            fmap_neibor=self.backbone.feat_net(image_neibor)  # 3*[b,32,44/22/11,44/22/11]
            fmap_mem=self.backbone.feat_net(image_mem)
            # b,c,h,w = fmap_cur[0].shape

            ## long-term
            keyQ, valueQ = self.membranch(fmap_cur)
            keyM, valueM00= self.membranch(fmap_mem)
            maskM = []
            for i in range(len(fmap_mem)):
                mask = F.interpolate(gt_mem, scale_factor=1/(8*2**i), mode='bilinear')
                maskM.append(mask)
            valueM = [valueM00[i].flatten(start_dim=-2)*(F.softmax(maskM[i].flatten(start_dim=-2),dim=-1)+1) for i in range(3)] 

            corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ,keyM,valueM)

            ## short-term
            # align
            fmap_ref =[x.clone() for x in fmap_cur]
            falign1=self.pcd_align(corr_vol_mem,fmap_ref)
            falign2=self.pcd_align(fmap_neibor,fmap_ref)
            # fuse
            corr_vol_neibor = self.nlnet(falign1, falign2)  # 3*[b,32,44/22/11,44/22/11]
        
            out = self.backbone.decoder(corr_vol_neibor)
            # seperate mask of different scale 
            mask_mem = out[-1][::-1]
            out = out[:-1]

#_____________________________________________________
        ## eval
        else:
            image1, image2 = x[0],x[1]
            fmap1=self.backbone.feat_net(image1)
            fmap2=self.backbone.feat_net(image2)  # 3*[b,32,44/22/11,44/22/11]
            # b,c,h,w = fmap1[0].shape

            ## long-term ##
            # val memory read
            keyQ, valueQ = self.membranch(fmap1)

            if use_mem:
                corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ)

            else:
                corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ,keyM_outer=keyQ,valueM_outer=valueQ)
                
            ## short-term ##
            # align
            falign1=self.pcd_align(corr_vol_mem,fmap1)
            falign2=self.pcd_align(fmap2,fmap1)
            # fuse
            corr_vol_neibor = self.nlnet(falign1, falign2)  # 3*[b,32,44/22/11,44/22/11]

            ## decode ##
            out = self.backbone.decoder(corr_vol_neibor)
            mask_mem = out[-1][::-1]
            out = out[:-1]

            ## memory load(should after memory read) ##
            if 0 in case_idx: 
                self.mem_bank.add_memory(keyQ, valueQ, mask_mem)
            elif add_mem:
   
                self.mem_bank.add_memory(keyQ, valueQ, mask_mem)
     
            # res = out[-1].sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # foreground = res[np.where(res>0.5)]
            # fore_conf=(2*(foreground - 0.5).sum() / len(foreground)) 
            # if 0 in case_idx: 
            #     self.mem_bank.add_memory(keyQ, valueQ, mask_mem)
            # elif add_mem and (fore_conf>=0.85): 
            #     self.mem_bank.add_memory(keyQ, valueQ, mask_mem)

        return out










    # def forward(self, x, case_idx, use_mem=False, add_mem=False, mode='train'):
    #     image1, image2 = x[0],x[1]
    #     fmap1=self.backbone.feat_net(image1)
    #     fmap2=self.backbone.feat_net(image2)  # 3*[b,32,44/22/11,44/22/11]
    #     b,c,h,w = fmap1[0].shape
    #     ## align
    #     fmap_ref =[x.clone() for x in fmap1]
    #     falign1=self.pcd_align(fmap1,fmap_ref)
    #     falign2=self.pcd_align(fmap2,fmap_ref)
 
    #     ## fuse&decode
    #     corr_vol_neibor = self.nlnet(falign1, falign2)  # 3*[b,32,44/22/11,44/22/11]
    #     out_neibor = self.backbone.decoder(corr_vol_neibor)

    #     ## seperate mask of different scale 
    #     mask_mem_neibor= out_neibor[-1][::-1]
    #     out_neibor = out_neibor[:-1]

    #     # valid when meet the first frame of this sequence
    #     if 0 in case_idx and mode == 'train':
    #         fmap_mem = [x[0].unsqueeze(0).clone().detach() for x in fmap1]
    #         mask_mem = []
    #         for i in range(len(fmap_mem)):
    #             mask = F.interpolate(self.first_case_gt, scale_factor=1/(8*2**i), mode='bilinear').clone().detach()
    #             mask_mem.append(mask)

    #         # fmap_mem_value = self.mask_attention(fmap_mem,mask_mem)
    #         # self.mem_bank.add_memory(fmap_mem,fmap_mem_value)
    #         self.mem_bank.add_memory(fmap_mem,mask_mem)
    #         self.first_case_gt = None

    #     # memory read
    #     if use_mem:
    #         fmap_mem = self.mem_bank.match_memory(corr_vol_neibor)
    #         falign_mem = self.pcd_align(fmap_mem,fmap_ref)
    #         corr_vol_mem = self.nlnet(falign1,falign_mem)
    #         out_mem = self.backbone.decoder(corr_vol_mem)
    #         # out_mem = self.backbone.decoder(fmap_mem)
    #         out_mem = out_mem[:-1]
    #         concated_mem = torch.cat([out_neibor[-1], out_mem[-1]], dim=1)
    #         out_final = self.fusion_conv(concated_mem) 

    #     # memory load(should after memory read)
    #     if add_mem:
            
    #         load_idx = random.randint(0,b-1)
    #         fmap_mem = [x[load_idx].unsqueeze(0).clone().detach() for x in fmap1]  # select random frame from the batch
    #         mask_mem = [x[load_idx].unsqueeze(0).clone().detach() for x in mask_mem_neibor]
    #         # fmap_mem_value = self.mask_attention(fmap_mem,mask_mem)
    #         # self.mem_bank.add_memory(fmap_mem,fmap_mem_value) 
    #         self.mem_bank.add_memory(fmap_mem,mask_mem)

    #     if use_mem:
    #         return out_neibor, out_mem, out_final
    #     else: 
    #         return out_neibor


## feature align module
class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        # self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        # self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.cas_dcnpack = DCNv2Pack(
        #     num_feat,
        #     num_feat,
        #     3,
        #     padding=1,
        #     deformable_groups=deformable_groups)

        # upsamplt and relu
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        feature_pyramid = []
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i-1], ref_feat_l[i-1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i-1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

            feature_pyramid.append(feat)
        # Cascading
        # offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        # offset = self.lrelu(
        #     self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        # feat = self.lrelu(self.cas_dcnpack(feat, offset))
        feature_pyramid=feature_pyramid[::-1]
        return feature_pyramid



## FFT correlation fusion
class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        Gx = torch.norm(inputs, p=2, dim=(2, 3), keepdim=True) # [1, 1, 1, c]
        x = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class FAM(nn.Module):
    def __init__(self, in_channels):
        super(FAM, self).__init__()
        self.se_block1 = SEBlock(in_channels//3 * 4, in_channels//3*2)
        self.se_block2 = SEBlock(in_channels//3 * 4, in_channels//3*2)

        self.reduce = nn.Conv2d(in_channels//3*2, in_channels//3, 3, 1, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels//3)
    
    def shift(self, x, n_segment=3, fold_div=8):
        z = torch.chunk(x, 3, dim=1)
        x = torch.stack(z,dim=1)
        b, nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(b, n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :, :-1, :fold] = x[:, :, 1:, :fold]  # shift left
        out[:, :, 1:, fold: 2 * fold] = x[:, :, :-1, fold: 2 * fold]  # shift right
        out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift

        return out.view(b, nt*c, h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        # temporal shift
        x = self.shift(x)
        # import pdb; pdb.set_trace()
        y = torch.fft.fft2(x)
        y_imag = y.imag
        y_real = y.real

        y1_imag, y2_imag, y3_imag = torch.chunk(y_imag, 3, dim=1)
        y1_real, y2_real, y3_real = torch.chunk(y_real, 3, dim=1)

        # grouping
        pair1 = torch.concat([y1_imag, y2_imag, y1_real, y2_real], dim=1)
        pair2 = torch.concat([y1_imag, y3_imag, y1_real, y3_real], dim=1)

        pair1 = self.se_block1(pair1).float()
        pair2 = self.se_block2(pair2).float()

        y1_real, y1_imag = torch.chunk(pair1, 2, dim=1)
        y1 = torch.complex(y1_real, y1_imag)
        z1 = torch.fft.ifft2(y1, s=(h, w)).float()

        y2_real, y2_imag = torch.chunk(pair2, 2, dim=1)
        y2 = torch.complex(y2_real, y2_imag)
        z2 = torch.fft.ifft2(y2, s=(h, w)).float()
        
        out = self.reduce(z1 + z2)
        out = F.relu(out)
        out = self.norm(out)
        
        return out