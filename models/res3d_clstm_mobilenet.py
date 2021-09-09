import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from models.resnext import resnext101 as resnext
from models.conv_lstm import ConvLSTM
from models.mobilenetv2_2d import mobilenetv2


class Res3D_cLSTM_MobileNet(nn.Module):
    
    def __init__(num_classes=num_classes, sample_size=112, sample_duration=16, res3d_out_size=256):
        # the author reshape input in shape (batchs, frames, width, height, channels) instead of (batchs, frames, channels, width, height)
        self.res3d = resnext(num_classes=res3d_out_size, sample_size=112, sample_duration=32, out_module_size=64)
        # res3d return output with size 2048, while the authors use 256
        # before the batch_first is set to False
        self.clstm2d_1 = ConvLSTM(256, hidden_dim, (3, 3), num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.clstm2d_2 = ConvLSTM(256, hidden_dim, (3, 3), num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.mobilenet = mobilenetv2(pretrained=True, num_classes=num_classes)
        self.global_pooling = nn.AvgPool3d(kernel_size=(sample_duration/2, 4, 4, 1024), stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.classifier = nn.Linear(out_features=num_classes)
        
        # Length × Width × Height × Channel format.
        
    
    def forward(self, x):
        x = self.res3d(x)
        x = self.clstm2d_1(x)
        x = self.clstm2d_2(x)
        x = torch.reshape(x, (28, 28, 256))
        x = mobilenet(x)
        x = global_pooling(x)
        x = torch.flatten(x)
        x = classifier(x)
        # add softmax
        
        return x