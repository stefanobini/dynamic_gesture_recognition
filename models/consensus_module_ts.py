import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torchinfo import summary
# from torchsummary import summary
#'''
from models.TimeSformer.timesformer.models.vit import TimeSformer

class ConsensusModuleTS(nn.Module):
    def __init__(self, num_classes=249, n_finetune_classes=249, sample_size=112, sample_duration=16, mod_aggr='avg', net=TimeSformer, modalities=['RGB'], feat_fusion=False, **kwargs):
        super(ConsensusModuleTS, self).__init__()
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.n_finetune_classes = n_finetune_classes
        self.modalities = modalities
        
        self.nets = nn.ModuleList()
        for modality in self.modalities:
            network = net(num_classes=num_classes, img_size=sample_size, num_frames=sample_duration, attention_type='divided_space_time')
            self.nets.append(network)
        
        self.mod_aggr = mod_aggr
        self.aggregator = None
        assert(self.mod_aggr in ['MLP', 'avg', 'max', 'none'])
        
        ########## Level in which fuse modalities ##########
        self.feat_dim = self.nets[0].get_classifier().in_features  if feat_fusion else self.num_classes
        
        if self.mod_aggr == 'MLP':
            self.aggregator = nn.Sequential(
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.feat_dim * len(self.modalities), self.num_classes)
            )
    
    
    def forward(self, x: Tensor) -> Tensor:
        # print('ConsensusModuleTS input shape: {}'.format(x.size()))  # (16, 1, 16, 3, 112, 112)
        nets_outputs = list()
        nets_features = list()
        if self.mod_aggr == 'MLP':
            # iterate on the modalities
            for i in range(x.size(1)):
                net_output = self.nets[i](x[:, i, :, :, :, :])
                nets_outputs.append(net_output[0])
                nets_features.append(net_output[1])
            x = torch.cat(nets_outputs, dim=1)
            x = self.aggregator(x)
        elif self.mod_aggr == 'avg':
            for i in range(x.size(1)):
                net_output = self.nets[i](x[:, i, :, :, :, :])
                nets_outputs.append(net_output[0])
                nets_features.append(net_output[1])
            x = torch.stack(nets_outputs, dim=1)
            x = x.mean(dim=1)
        elif self.mod_aggr == 'max':
            for i in range(x.size(1)):
                net_output = self.nets[i](x[:, i, :, :, :, :])
                nets_outputs.append(net_output[0])
                nets_features.append(net_output[1])
            x = torch.stack(nets_outputs, dim=1)
            x = x.max(dim=1)
        elif self.mod_aggr == 'none':  # the case in which use a single modality
            x = x[:, 0, :, :, :, :].permute(0, 2, 1, 3, 4)
            # print('TimeSformer input shape: {}'.format(x[:, 0, :, :, :, :].size()))
            x = self.nets[0](x)
            nets_outputs = x
        else:
            x = None
            nets_outputs = None
            nets_features = None
        return x, nets_outputs, nets_features

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    
    elif ft_portion == "last_layer":
        for param in model.module.parameters():
            param.requires_grad = False
        for net in model.module.nets:
            for param in net.get_classifier().parameters():
                param.requires_grad = True
        if model.module.aggregator is not None:
            for param in model.module.aggregator.parameters():
                param.requires_grad = True
        return model.parameters()
        
    elif ft_portion == "aggregator":
        for param in model.module.parameters():
            param.requires_grad = False
        if model.module.aggregator is not None:
            for param in model.module.aggregator.parameters():
                param.requires_grad = True
        return model.parameters()
        
    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(net='timesformer', *args, **kwargs):
    model = ConsensusModuleTS(net=TimeSformer, *args, **kwargs)
    
    return model

    
if __name__ == "__main__":
    kwargs = dict()
    model = get_model(num_classes=249, sample_size=112, sample_duration=16, net='transformer', mod_aggr='none', modalities=['RGB'], feat_fusion=False)
    # print(model)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    input_shape = (1, 3, 16, 112, 112)
    # model_sum = summary(model.module, input_shape)
    input_var = Variable(torch.randn(input_shape))
    print('Input var shape: {}'.format(input_var.shape))
    input_list = [input_var for i in range(16)]
    input_tensor = torch.stack(input_list)
    print('Shape of input tensor: {}'.format(input_var.shape))
    # print('Lenght input list: {}'.format(len(input_list)))
    # print('Shape of list element: {}'.format(input_list[0].shape))


    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_tensor)
    print('Output shape: {}'.format(output.size()))