import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary


class Res3D(nn.Module):

    def __init__(self):
        
        super(Res3D, self).__init__()
        
        # Res3D Block 1
        self.conv3d_1 = nn.Sequential(nn.Conv3d(3, 64, (3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True))

        # Res3D Block 2
        self.conv3d_2a_1 = nn.Sequential(nn.Conv3d(64, 64, (1,1,1), padding='same', bias=False),
                                        nn.BatchNorm3d(64))
        self.conv3d_2a_a = nn.Sequential(nn.Conv3d(64, 64, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(inplace=True))
        self.conv3d_2a_b = nn.Sequential(nn.Conv3d(64, 64, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(64))
        self.conv3d_2a = nn.ReLU(inplace=True)
        
        self.conv3d_2b_a = nn.Sequential(nn.Conv3d(64, 64, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(inplace=True))       
        self.conv3d_2b_b = nn.Sequential(nn.Conv3d(64, 64, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(64))
        self.conv3d_2b = nn.ReLU(inplace=True)

        # Res3D Block 3
        self.conv3d_3a_1 = nn.Sequential(#nn.ConstantPad3d((26,26,26,26,7,7), 27),
                                        nn.Conv3d(64, 128, (1,1,1), stride=(2,2,2), bias=False),
                                        nn.BatchNorm3d(128))       
        self.conv3d_3a_a = nn.Sequential(#nn.ConstantPad3d((27,27,27,27,8,8), 27),
                                        nn.Conv3d(64, 128, (3,3,3), stride=(2,2,2), padding=1, bias=False),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(inplace=True))
        # adapt in order to obtain conv3d_3a_1.size() == conv3d_3a_b.size()
        self.conv3d_3a_b = nn.Sequential(nn.Conv3d(128, 128, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(128))
        self.conv3d_3a = nn.ReLU(inplace=True)

        self.conv3d_3b_a = nn.Sequential(nn.Conv3d(128, 128, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(inplace=True))
        self.conv3d_3b_b = nn.Sequential(nn.Conv3d(128, 128, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(128))
        self.conv3d_3b = nn.ReLU(inplace=True)

        # Res3D Block 4
        self.conv3d_4a_1 = nn.Sequential(nn.Conv3d(128, 256, (1,1,1), padding='same', bias=False),
                                        nn.BatchNorm3d(256))
        self.conv3d_4a_a = nn.Sequential(nn.Conv3d(128, 256, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(256),
                                        nn.ReLU(inplace=True))
        self.conv3d_4a_b = nn.Sequential(nn.Conv3d(256, 256, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(256))
        self.conv3d_4a = nn.ReLU(inplace=True)

        self.conv3d_4b_a = nn.Sequential(nn.Conv3d(256, 256, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(256),
                                        nn.ReLU(inplace=True))
        self.conv3d_4b_b = nn.Sequential(nn.Conv3d(256, 256, (3,3,3), padding='same', bias=False),
                                        nn.BatchNorm3d(256))
        self.conv3d_4b = nn.ReLU(inplace=True)
    
    
    def forward(self, input):
        # Res3D Block 2
        conv3d_1 = self.conv3d_1(input)

        # Res3D Block 2
        conv3d_2a_1 = self.conv3d_2a_1(conv3d_1)
        conv3d_2a_a = self.conv3d_2a_a(conv3d_2a_1)
        conv3d_2a_b = self.conv3d_2a_b(conv3d_2a_a)
        # print('conv3d_2a_1: {}\nconv3d_2a_a: {}\nconv3d_2a_b: {}\n'.format(conv3d_2a_1.size(), conv3d_2a_a.size(), conv3d_2a_b.size()))
        conv3d_2a = torch.add(conv3d_2a_1, conv3d_2a_b)
        conv3d_2a = self.conv3d_2a(conv3d_2a)

        conv3d_2b_a = self.conv3d_2b_a(conv3d_2a)
        conv3d_2b_b = self.conv3d_2b_b(conv3d_2b_a)
        # print('conv3d_2a: {}\nconv3d_2b_a: {}\nconv3d_2b_b: {}\n'.format(conv3d_2a.size(), conv3d_2b_a.size(), conv3d_2b_b.size()))
        conv3d_2b = torch.add(conv3d_2a, conv3d_2b_b)
        conv3d_2b = self.conv3d_2b(conv3d_2b)

        # Res3D Block 3
        conv3d_3a_1 = self.conv3d_3a_1(conv3d_2b)
        conv3d_3a_a = self.conv3d_3a_a(conv3d_2b)
        conv3d_3a_b = self.conv3d_3a_b(conv3d_3a_a)
        # print('conv3d_3a_1: {}\nconv3d_3a_a: {}\nconv3d_3a_b: {}\n'.format(conv3d_3a_1.size(), conv3d_3a_a.size(), conv3d_3a_b.size()))
        conv3d_3a = torch.add(conv3d_3a_1, conv3d_3a_b)
        conv3d_3a = self.conv3d_3a(conv3d_3a)

        conv3d_3b_a = self.conv3d_3b_a(conv3d_3a)
        conv3d_3b_b = self.conv3d_3b_b(conv3d_3b_a)
        # print('conv3d_3a: {}\nconv3d_3b_a: {}\nconv3d_3b_b: {}\n'.format(conv3d_3a.size(), conv3d_3b_a.size(), conv3d_3b_b.size()))
        conv3d_3b = torch.add(conv3d_3a, conv3d_3b_b)
        conv3d_3b = self.conv3d_3b(conv3d_3b)

        # Res3D Block 4
        conv3d_4a_1 = self.conv3d_4a_1(conv3d_3b)
        conv3d_4a_a = self.conv3d_4a_a(conv3d_3b)
        conv3d_4a_b = self.conv3d_4a_b(conv3d_4a_a)
        # print('conv3d_4a_1: {}\nconv3d_4a_a: {}\nconv3d_4a_b: {}\n'.format(conv3d_4a_1.size(), conv3d_4a_a.size(), conv3d_4a_b.size()))
        conv3d_4a = torch.add(conv3d_4a_1, conv3d_4a_b)
        conv3d_4a = self.conv3d_4a(conv3d_4a)

        conv3d_4b_a = self.conv3d_4b_a(conv3d_4a)
        conv3d_4b_b = self.conv3d_4b_b(conv3d_4b_a)
        # print('conv3d_4a: {}\nconv3d_4b_a: {}\nconv3d_4b_b: {}\n'.format(conv3d_4a.size(), conv3d_4b_a.size(), conv3d_4b_b.size()))
        conv3d_4b = torch.add(conv3d_4a, conv3d_4b_b)
        conv3d_4b = self.conv3d_4b(conv3d_4b)
        
        return conv3d_4b


if __name__ == "__main__":
    kwargs = dict()
    model = Res3D()
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
    output = model(input_tensor)
    print('Output shape: {}\nPermuted output shape: {}'.format(output.size(), output.permute(0, 2, 1, 3, 4).size()))