import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary


class SeparableConv2d(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv2d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv2d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        x = self.spatialConv(x)
        x = self.pointConv(x)
        return x


class MobileNet(nn.Module):
    def __init__(self):
        
        super(MobileNet, self).__init__()
        
        self.conv2d_1a = nn.Sequential(SeparableConv2d(256, 256, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU6())

        self.conv2d_1b = nn.Sequential(SeparableConv2d(256, 256, (3,3), stride=(2,2), padding=15),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU6())

        self.conv2d_2a = nn.Sequential(SeparableConv2d(256, 512, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU6())

        self.conv2d_2b = nn.Sequential(SeparableConv2d(512, 512, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU6())

        self.conv2d_2c = nn.Sequential(SeparableConv2d(512, 512, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU6())

        self.conv2d_2d = nn.Sequential(SeparableConv2d(512, 512, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU6())

        self.conv2d_2e = nn.Sequential(SeparableConv2d(512, 512, (3,3), stride=(1,1), padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU6())

        self.conv2d_3a = nn.Sequential(SeparableConv2d(512, 1024, (3,3), stride=(2,2), padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU6())

        self.conv2d_3b = nn.Sequential(SeparableConv2d(1024, 1024, (3,3), stride=(2,2), padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU6())
        
        
    def forward(self, x):
        conv2d_1a = self.conv2d_1a(x)
        # print('conv2d_1a: {}'.format(conv2d_1a.size()))
        conv2d_1b = self.conv2d_1b(conv2d_1a)
        # print('conv2d_1b: {}'.format(conv2d_1b.size()))
        conv2d_2a = self.conv2d_2a(conv2d_1b)
        # print('conv2d_2a: {}'.format(conv2d_2a.size()))
        conv2d_2b = self.conv2d_2b(conv2d_2a)
        # print('conv2d_2b: {}'.format(conv2d_2b.size()))
        conv2d_2c = self.conv2d_2c(conv2d_2b)
        # print('conv2d_2c: {}'.format(conv2d_2c.size()))
        conv2d_2d = self.conv2d_2d(conv2d_2c)
        # print('conv2d_2d: {}'.format(conv2d_2d.size()))
        conv2d_2e = self.conv2d_2e(conv2d_2d)
        # print('conv2d_2e: {}'.format(conv2d_2e.size()))
        conv2d_3a = self.conv2d_3a(conv2d_2e)
        # print('conv2d_3a: {}'.format(conv2d_3a.size()))
        conv2d_3b = self.conv2d_3b(conv2d_3a)
        # print('conv2d_3b: {}'.format(conv2d_3b.size()))
    
        return conv2d_3b


if __name__ == "__main__":
    kwargs = dict()
    model = MobileNet()
    # print(model)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    input_shape = (256, 28, 28)
    #model_sum = summary(model.module, input_shape)
    input_var = Variable(torch.randn(input_shape))
    print('Input var shape: {}'.format(input_var.size()))
    input_list = [input_var for i in range(16)]
    input_tensor = torch.stack(input_list)
    print('Shape of input tensor: {}'.format(input_tensor.size()))
    # print('Lenght input list: {}'.format(len(input_list)))
    # print('Shape of list element: {}'.format(input_list[0].shape))


    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_tensor)
    print('Output shape: {}'.format(output.size()))