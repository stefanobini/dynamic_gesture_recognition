import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torchinfo import summary
# from torchsummary import summary

from models.mobilenetv2_2d import mobilenetv2
from models.resnext_2d import resnext101_32x8d


class ConsensusModule2DCNN(nn.Module):
    def __init__(self, num_classes=249, n_finetune_classes=249, sample_size=112, sample_duration=16, modalities=['RGB'], mod_aggr='none', temp_aggr='avg', net=mobilenetv2, width_mult=1., groups=3, width_per_group=32):
        super(ConsensusModule2DCNN, self).__init__()
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.n_finetune_classes = n_finetune_classes
        self.modalities = modalities
        
        self.temp_aggr = temp_aggr
        assert(self.temp_aggr in ['MLP', 'LSTM', 'avg', 'max'])
        self.mod_aggr = mod_aggr
        assert(self.mod_aggr in ['MLP', 'avg', 'max', 'none'])
        self.mod_aggregator = None
        
        self.temp_aggregators = nn.ModuleList()
        self.mod_nets = nn.ModuleList()     # sets of 2D-CNNs, one for each modality, each set is composed by a set of 2D-CNNs one for each frame
        for mod in range(len(modalities)):
            temp_nets = nn.ModuleList()
            temp_aggregator = None
            for frame in range(self.sample_duration):
                cnn = nn.Sequential(
                    # nn.AdaptiveAvgPool2d((32, 32)),
                    net(pretrained=True, num_classes=num_classes))
                temp_nets.append(cnn)
                
                # Adding temporal aggregator
                if self.temp_aggr == 'MLP':
                    temp_aggregator = nn.Sequential(
                        # nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(self.num_classes * self.sample_duration, self.num_classes)
                    )
                    self.temp_aggregators.append(temp_aggregator)
                elif self.temp_aggr == 'LSTM':
                    temp_aggregator = nn.LSTM(input_size=self.num_classes, hidden_size=self.num_classes, batch_first=False, bidirectional=True)
                    self.temp_aggregators.append(temp_aggregator)
            
            self.mod_nets.append(temp_nets)
         
        if self.mod_aggr == 'MLP':
            self.mod_aggregator = nn.Sequential(
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.num_classes * len(self.modalities), self.num_classes)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        # print('0 - INPUT SAMPLE: ', x.size())
        cnns_outputs = list()
        for i in range(x.size(1)):
            if self.temp_aggr == 'MLP':
                # iterate on the frames
                # print('1 - cons_2dcnn x size: ', x[:, i, 0, :, :, :].size())
                cnn_output = torch.cat([self.mod_nets[i][j](x[:, i, j, :, :, :]) for j in range(x.size(2))], dim=1)  # self.cnns[i](x[:, 0, i, :, :, :])[0], have insert [0], because the forward of Mobilenet2d return also the features
                # print('2 - cons_2dcnn cnn size: ', cnn_output.size())
                cnn_output = self.temp_aggregators[i](cnn_output)
            elif self.temp_aggr == 'LSTM':
                h_0 = torch.randn(2, x.size(0), self.n_finetune_classes).cuda()
                c_0 = torch.randn(2, x.size(0), self.n_finetune_classes).cuda()
                for j in range(x.size(1)):
                    pred = self.mod_nets[i][j](x[:, i, j, :, :, :])
                    x1, (h_0, c_0) = self.temp_aggregators[i](pred.view(-1, x.size(0), self.n_finetune_classes), (h_0, c_0))
                cnn_output = x1[-1]
                cnn_output = cnn_output.view(cnn_output.size(0), -1, cnn_output.size(1))      # stack output of bidirectional cells, because it is doubled
                cnn_output = cnn_output.mean(dim=1)                                           # average the output of bidirectional cells
            elif self.temp_aggr == 'avg':
                cnn_output = torch.stack([self.mod_nets[i][j](x[:, i, j, :, :, :]) for j in range(x.size(2))], dim=1)
                cnn_output = cnn_output.mean(dim=1)
            elif self.temp_aggr == 'max':
                cnn_output = torch.stack([self.mod_nets[i][j](x[:, i, j, :, :, :]) for j in range(x.size(2))], dim=1)
                cnn_output = cnn_output.max(dim=1)
            else:
                cnn_output = None
            cnns_outputs.append(cnn_output)
        
        # print('3 - cons_2dcnn cnns size: {}\tlenght: {}'.format(cnns_outputs[0].size(), len(cnns_outputs)))
        if self.mod_aggr == 'MLP':
            x = torch.cat(cnns_outputs, dim=1)
            x = self.mod_aggregator(x)
        elif self.mod_aggr == 'avg':
            x = torch.stack(cnns_outputs, dim=1)
            x = x.mean(dim=1)
        elif self.mod_aggr == 'max':
            x = torch.stack(cnns_outputs, dim=1)
            x = x.max(dim=1)
        elif self.mod_aggr == 'none':  # the case in which use a single modality
            # print('4 - HEY')
            x = cnns_outputs[0]
            cnns_outputs = x
        else:
            x = None
            cnns_outputs = None
        # print('5 - cons_2dcnn X size: {}\tcnns_outputs: {}'.format(x.size(), cnns_outputs[0].size()))
        return x, cnns_outputs


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    
    elif ft_portion == "last_layer":
        for param in model.module.parameters():
            param.requires_grad = False
        for pool_nets in model.module.mod_nets:
            for net in pool_nets:
                for param in net[0].classifier.parameters():
                    param.requires_grad = True
        if model.module.mod_aggregator is not None:
            for param in model.module.mod_aggregator.parameters():
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