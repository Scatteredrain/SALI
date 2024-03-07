import torch
from torch import nn
from torch.nn import functional as F

class SAM(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, sub_sample_scale=2):
        super(SAM, self).__init__()  #(in_channels*2, inter_channels=None,
                                                    #  dimension=2, sub_sample=True, bn_layer=False, sub_sample_scale=2,4,8)
        
        self.sub_sample = sub_sample
        self.bn_layer = bn_layer
        self.sub_sample_scale = sub_sample_scale

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(sub_sample_scale, sub_sample_scale))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.residual_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

        

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)  [b,32*2,22,22]
        :return:
        '''
        output = [x]

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # g(x) -> fuse[conv] frames & downsample[maxpool]

        g_x = g_x.permute(0, 2, 1)       
        
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  #[1,22*22,32*2]

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)  # maxpool
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)       
        f_div_C = F.softmax(f, dim=-1)         

        y = torch.matmul(f_div_C, g_x)         
        y = y.permute(0, 2, 1).contiguous()   
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) 
        W_y = self.W(y)                                           
        z = W_y + x

        output.append(z)
        output = torch.cat(output,dim=1)
        output = self.residual_conv(output)

        return output
