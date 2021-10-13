import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
#'''
from models.res3d import Res3D
from models.conv_lstm import ConvLSTM
from models.mobilenet_custom import MobileNet
'''
from res3d import Res3D
from conv_lstm import ConvLSTM
from mobilenet_custom import MobileNet
'''

class Res3D_cLSTM_MobileNet(nn.Module):
    
    def __init__(self, num_classes=249, sample_size=112, sample_duration=16):
        super(Res3D_cLSTM_MobileNet, self).__init__()
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        # the author reshape input in shape (batches, frames, width, height, channels) instead of (batches, frames, channels, width, height)
        self.res3d = Res3D()
        # before the batch_first is set to False
        self.clstm2d = ConvLSTM(input_dim=256, hidden_dim=256, kernel_size=(3,3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        
        self.mobilenet_module = nn.ModuleList()
        for i in range(self.sample_duration):
            mobilenet = MobileNet()
            self.mobilenet_module.append(mobilenet)
        
        self.global_pooling = nn.AvgPool3d(kernel_size=(int(sample_duration/2), 7, 7), stride=(int(self.sample_duration/2), 7, 7), padding=1, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        
    
    def forward(self, x):
        # print('Res3D_cLSTM_MobileNet input shape: {}'.format(x.size()))     # (3, 16, 112, 112)
        x = self.res3d(x)
        # print('Res3D output shape: ', x.size())
        x = x.permute(0, 2, 1, 3, 4)
        # print('Res3D output reshape: ', x.size())
        x, _ = self.clstm2d(x)
        #x = x[0].permute(0, 1, 3, 4, 2)
        #x = torch.reshape(x[0], (28, 28, 256))
        x = x[0]
        # print('ConvLSTM output shape: ', x.size())
        x = torch.stack([self.mobilenet_module[i](x[:, i, :, :, :]) for i in range(x.size()[1])], dim=1)
        #x = torch.stack([self.mobilenet_module[i](x[:, 0, i, :, :, :])[1] for i in range(x.size()[1])])
        #_, x = self.mobilenet(x[0])
        
        # print('MobileNet output shape: {}'.format(x[0].size()))
        # print('MobileNets output shape: {}'.format(x.size()))
        x = x.permute(0, 2, 1, 3, 4)     # (int(self.sample_duration/2), 4, 4, 1024)
        # print('Reshape tensor, before global pooling: {}'.format(x.size()))
        x = self.global_pooling(x)
        # print('Global average pooling shape: {}'.format(x.size()))
        
        features = torch.squeeze(x)
        # print('Squeeze output shape: ', x.size())
        x = self.classifier(features)
        # print('Classifier output shape: ', x.size())
        #add softmax
        
        return x, features
        

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


if __name__ == "__main__":
    kwargs = dict()
    model = Res3D_cLSTM_MobileNet(num_classes=249, sample_size=112, sample_duration=16)
    # print(model)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    input_shape = (3, 16, 112, 112)
    #model_sum = summary(model.module, input_shape)
    input_var = Variable(torch.randn(input_shape))
    print('Input var shape: {}'.format(input_var.size()))
    input_list = [input_var for i in range(16)]
    input_tensor = torch.stack(input_list)
    print('Shape of input tensor: {}'.format(input_tensor.size()))
    # print('Lenght input list: {}'.format(len(input_list)))
    # print('Shape of list element: {}'.format(input_list[0].shape))


    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output, features = model(input_tensor)
    print('Output shape: {}'.format(output.size()))