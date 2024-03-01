import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def softmax_w_top(x, top, return_usage=False):
    top = min(top, x.shape[1]) # min(top,THW)
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = torch.softmax(values, dim=1)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    if return_usage:   
        return x, x.sum(dim=2)
    else:
        return x


class MemoryBank():                                                           
    def __init__(self, test_mem_length=35, num_values=3, top_k=20, count_usage=True):
        self.top_k = top_k                                                  

        self.CK = None
        self.CV = None
        self.num_values = num_values
        self.test_mem_length = test_mem_length
        # self.test_mem_size = 10000

        self.mem_ks = [None for i in range(self.num_values)]                                             
        self.mem_vs = [None for i in range(self.num_values)]
        self.T = 0

        self.CountUsage = count_usage
        if self.CountUsage:
            self.usage_count = [None for i in range(self.num_values)]
            self.life_count = [None for i in range(self.num_values)]

    # def _mask_attention(self,feat,mask):
    #     f_att = [x.clone().detach() for x in feat]
    #     for i in range(len(feat)):
    #         b,c,_ = mask[i].shape
    #         s = F.softmax(mask[i],dim=-1).view(b,c,-1)
    #         f_att[i] = feat[i]*s + feat[i]

    #     return f_att

    def _global_matching(self, mk, qk): 
        # mk:[c,h*w*t]; qk:[b,c,h*w]
        # NE means number of elements -- typically T*H*W                                    
        B, CK, NE = mk.shape                                                
                                                                            
        a = mk.pow(2).sum(1).unsqueeze(2)   #[b,hw,1]                        
        b = 2 * (mk.transpose(1, 2) @ qk)   #[b,thw,hw]                                  

        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW; [B, [256|512|...], 256]
        # if self.training:
        #     maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        #     x_exp = torch.exp(affinity - maxes)
        #     x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        #     affinity = x_exp / x_exp_sum 
        # else:
        
        return affinity
                                                                            
    def _readout(self, affinity, value_mem, value_query):   
        # affinity:[b,hw,thw];   value_mem:[1,32/2,thw]
        # value_query:[1,32/2,h,w] ; return:[b,32,thw]
        b, c_k, h, w = value_query.shape
        return torch.cat([torch.bmm(value_mem,affinity).view(b,-1,h,w),value_query],dim=1)                                 
                                                                            
    def match_memory(self, keyQ, valueQ, keyM_outer=None, valueM_outer=None):
        readout_mems = []

        if (keyM_outer is not None) and (valueM_outer is not None):
            if len(keyM_outer[0].shape) == 4:
                keyM = [x.flatten(start_dim=-2) for x in keyM_outer] 
            else:
                keyM = keyM_outer
            
            if len(valueM_outer[0].shape) == 4:
                valueM = [x.flatten(start_dim=-2) for x in valueM_outer] 
            else:
                valueM = valueM_outer

        else:
            keyM,valueM = self.mem_ks,self.mem_vs

        for i in range(self.num_values):                                        
                
            key_query, value_query = keyQ[i], valueQ[i]
            key_mem, value_mem = keyM[i], valueM[i]
            
            affinity = self._global_matching(key_mem, key_query.flatten(start_dim=-2))

            ## k-filter
            if self.CountUsage and (None not in self.mem_ks):
                affinity, usage_new = softmax_w_top(affinity, top=self.top_k,return_usage=True)  # B, NE, HW    
                self.update_usage(usage_new=usage_new,scale=i)
            else:
                affinity = softmax_w_top(affinity, top=self.top_k,return_usage=False)  # B, NE, HW    
            readout_mems.append(self._readout(affinity, value_mem, value_query))

        return readout_mems

    def add_memory(self,keyQ_mem,valueQ_mem,mask_mem):


        # keys: 3*[b,32,44/22/11,44/22/11]
        keys_mem = [keyQ_mem[i].flatten(start_dim=-2) 
                    for i in range(self.num_values)]
        values_mem = [valueQ_mem[i].flatten(start_dim=-2)*(F.softmax(mask_mem[i].flatten(start_dim=-2),dim=-1)+1) 
                      for i in range(self.num_values)] 
        if self.CountUsage:
            new_count = [torch.zeros((keys_mem[i].shape[0], 1, keys_mem[i].shape[2]), device=keys_mem[0].device, dtype=torch.float32) 
                            for i in range(self.num_values)]
            new_life = [torch.zeros((keys_mem[i].shape[0], 1, keys_mem[i].shape[2]), device=keys_mem[0].device, dtype=torch.float32) + 1e-7
                            for i in range(self.num_values)]
        
        # keys: 3*[b,32,h*w]
        if None in self.mem_ks :                                              
            self.mem_ks = keys_mem
            self.mem_vs = values_mem
            self.CK = keys_mem[0].shape[1]
            self.CV = values_mem[0].shape[1]
            self.hwK0 = keys_mem[0].shape[2]
            self.hwV0 = values_mem[0].shape[2]
            if self.CountUsage:
                self.usage_count = new_count
                self.life_count = new_life
            # self.T = 1
        else:                                                               
            self.mem_ks = [torch.cat([self.mem_ks[i], keys_mem[i]], 2) for i in range(self.num_values)]
            self.mem_vs = [torch.cat([self.mem_vs[i], values_mem[i]], 2) for i in range(self.num_values)]
            # self.T += 1
            # if (self.test_mem_length is not None):
            #     self.mem_ks = [self.mem_ks[i][..., -self.hwK0*self.test_mem_length:]  for i in range(self.num_values)]
            #     self.mem_vs = [self.mem_vs[i][..., -self.hwV0*self.test_mem_length:]  for i in range(self.num_values)]
            
            if self.CountUsage:
                self.usage_count = [torch.cat([self.usage_count[i], new_count[i]], -1) for i in range(self.num_values)]
                self.life_count =[torch.cat([self.life_count[i], new_life[i]], -1) for i in range(self.num_values)]
                if self.T >= self.test_mem_length:
                    self.obsolete_features_removing(self.test_mem_length)

        self.T = self.mem_ks[0].shape[2] // self.hwK0

    def clear_memory(self):
  
        self.mem_ks = [None for i in range(self.num_values)]
        self.mem_vs = [None for i in range(self.num_values)]
        if self.CountUsage:
            self.usage_count = [None for i in range(self.num_values)]
            self.life_count = [None for i in range(self.num_values)]
        self.T = 0
        # print('clear memory!')

    def update_usage(self, usage_new, scale):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.CountUsage:
            return
        
        self.usage_count[scale] += usage_new.view_as(self.usage_count[scale])
        self.life_count[scale] += 1

    def get_usage_scale(self,scale):
        # return normalized usage
        if not self.CountUsage:
            raise RuntimeError('No count usage!')
        else:
            usage = self.usage_count[scale] / self.life_count[scale]
            return usage

    def obsolete_features_removing(self, max_length: int):
        # normalize with life duration
        # B = 1
        for i in range(self.num_values):

            usage = self.get_usage_scale(scale=i).flatten()  #[B*T*H*W]
            max_size = max_length * (self.hwK0) * (1/4)**i
            # print('remove:{}'.format(str(int(self.size[i]-max_size))))
            values, index = torch.topk(usage, k=int(self.size[i]-max_size), largest=False, sorted=True)
            survived = (usage > values[-1])  

            self.mem_ks[i] = self.mem_ks[i][:, :, survived] if self.mem_ks[i] is not None else None
            self.mem_vs[i] = self.mem_vs[i][:, :, survived] if self.mem_vs[i] is not None else None
            self.usage_count[i] = self.usage_count[i][:, :, survived]
            self.life_count[i] = self.life_count[i][:, :, survived]

    @property
    def size(self):
        if self.mem_ks[0] is None:
            return [0 for i in range(self.num_values)]
        else:
            return [self.mem_ks[i].shape[-1] for i in range(self.num_values)] #T*H*W