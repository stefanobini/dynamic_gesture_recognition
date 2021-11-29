import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torchinfo import summary
# from torchsummary import summary

from models.mobilenetv2_2d import mobilenetv2
from models.resnext_2d import resnext101_32x8d


class Video_Transformer(nn.Module):
    def __init__(self, num_classes=249, n_finetune_classes=249, sample_size=112, sample_duration=16, modalities=['RGB'], mod_aggr='none', temp_aggr='avg', net=mobilenetv2, width_mult=1., groups=3, width_per_group=32):
        super(Video_Transformer, self).__init__()
        self.spatial_tran_name = spatial_tran_name
        
        self.spatial_feat_extractor = 
        self.spatial_tran = 
        self.temporal_tran = 
        
    
    def forward(self, x):
        spatial_feat = spatial_feat_extractor(x)
        
        spatial_emb = spatial_tran(spatial_feat)
        
        temporal_emb = temporal_tran(spatial_emb)
        