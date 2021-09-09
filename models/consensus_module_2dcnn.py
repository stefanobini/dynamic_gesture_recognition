import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torchinfo import summary
# from torchsummary import summary

from models.mobilenetv2_2d import mobilenetv2
from models.resnext_2d import resnext101_32x8d


class ConsensusModule2DCNN(nn.Module):
    def __init__(self, num_classes=249, n_finetune_classes=249, sample_size=112, width_mult=1., sample_duration=16, aggr_type='avg', net=mobilenetv2):
        super(ConsensusModule2DCNN, self).__init__()
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.n_finetune_classes = n_finetune_classes
        
        self.cnns = nn.ModuleList()
        for i in range(self.sample_duration):
            cnn = nn.Sequential(
                # nn.AdaptiveAvgPool2d((32, 32)),
                net(pretrained=True, num_classes=num_classes))
            self.cnns.append(cnn)
        
        self.aggr_type = aggr_type
        self.aggregator = None
        assert(self.aggr_type in ['MLP', 'LSTM', 'avg'])
        if self.aggr_type == 'MLP':
            self.aggregator = nn.Sequential(
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.num_classes * self.sample_duration, self.num_classes)
            )
        elif self.aggr_type == 'LSTM':
            self.aggregator = nn.LSTM(input_size=self.num_classes, hidden_size=self.num_classes, batch_first=False, bidirectional=True)
    
    def forward(self, x: Tensor) -> Tensor:
        # print('INPUT SAMPLE: ', x.size())
        if self.aggr_type == 'MLP':
            # iterate on the frames
            x = torch.cat([self.cnns[i](x[:, 0, i, :, :, :]) for i in range(x.size(1))], dim=1)
            x = self.aggregator(x)
        elif self.aggr_type == 'LSTM':
            h_0 = torch.randn(2, x.size(0), self.n_finetune_classes).cuda()
            c_0 = torch.randn(2, x.size(0), self.n_finetune_classes).cuda()
            for i in range(x.size(1)):
                pred = self.cnns[i](x[:, 0, i, :, :, :])
                x1, (h_0, c_0) = self.aggregator(pred.view(-1, x.size(0), self.n_finetune_classes), (h_0, c_0))
            x = x1[-1]
            x = x.view(x.size(0), -1, x.size(1))      # stack output of bidirectional cells, because it is doubled
            x = x.mean(dim=1)                                           # average the output of bidirectional cells
        elif self.aggr_type == 'avg':
            x = torch.stack([self.cnns[i](x[:, 0, i, :, :, :]) for i in range(x.size()[1])])
            x = x.mean(dim=0)
        else:
            x = None
        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    
    elif ft_portion == "last_layer":
        for param in model.module.parameters():
            param.requires_grad = False
        for net in model.module.cnns:
            for param in net[0].classifier.parameters():
                param.requires_grad = True
        if model.module.aggregator is not None:
            for param in model.module.aggregator.parameters():
                param.requires_grad = True
        
        return model.parameters()
    
    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
    '''
    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')
        ft_module_names.append('aggregator')
        
        count_1 = 0
        count_2 = 0
        parameters = []
        for k, v in model.named_parameters():
            # count_1 += 1
            for ft_module in ft_module_names:
                if ft_module in k:
                    count_1 += 1
                    # print('######### BEFORE BREAK! #########')
                    # parameters.append({'params': v})
                    parameters.append({'params': v, 'requires_grad': True})
                    # print('* Learning rate: {} *'.format(parameters[-1].keys()))
                    break
            else:
                count_2 += 1
                # print('######### IN ELSE! #########')
                # parameters.append({'params': v, 'lr': 0.0})
                parameters.append({'params': v, 'requires_grad': False})
                # print('* Learning rate: {} *'.format(parameters[-1].keys()))
        print('*****\nCount_1: {}\nCount_2 : {}\n*****'.format(count_1, count_2))
        return parameters
        
    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
    '''


def get_model(net='mobilenetv2_2d', **kwargs):
    if net == 'mobilenetv2_2d': 
        model = ConsensusModule2DCNN(net=mobilenetv2, **kwargs)
    elif net == 'resnext_2d':
        model = ConsensusModule2DCNN(net=resnext101_32x8d, **kwargs)
    return model

    
if __name__ == "__main__":
    kwargs = dict()
    model = get_model(num_classes=249, sample_size=112, width_mult=1., net='mobilenetv2_2d')
    # print(model)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    input_shape = (32, 8, 3, 112, 112)
    model_sum = summary(model.module, input_shape)
    input_var = Variable(torch.randn(input_shape))
    print('Input var shape: {}'.format(input_var.shape))
    input_list = [input_var for i in range(16)]
    input_tensor = torch.stack(input_list)
    print('Shape of input tensor: {}'.format(input_var.shape))
    # print('Lenght input list: {}'.format(len(input_list)))
    # print('Shape of list element: {}'.format(input_list[0].shape))


    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_tensor)
    print(output.shape)