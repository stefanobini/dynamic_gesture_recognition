import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torchinfo import summary
# from torchsummary import summary
#'''
from models.mobilenetv2 import get_model as mobilenetv2
from models.resnext import resnext101 as resnext
from models.res3d_clstm_mobilenet import Res3D_cLSTM_MobileNet
from models import RAAR3DNet, NI3D
#from models.EAN_16f import resnet101 as transformer
'''
from mobilenetv2 import get_model as mobilenetv2
from resnext import resnext101 as resnext
import RAAR3DNet, NI3D
#from res3d_clstm_mobilenet import Res3D_cLSTM_MobileNet
from EAN_16f import resnet101 as transformer
#'''

class ConsensusModule3DCNN(nn.Module):
    def __init__(self, num_classes=249, n_finetune_classes=249, sample_size=112, sample_duration=16, mod_aggr='avg', net=resnext, modalities=['RGB'], feat_fusion=False, **kwargs):
        super(ConsensusModule3DCNN, self).__init__()
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.n_finetune_classes = n_finetune_classes
        self.modalities = modalities
        
        self.cnns = nn.ModuleList()
        for modality in self.modalities: 
            if net in [RAAR3DNet.Network, NI3D.Network]:
                cnn = net(args=kwargs, num_classes=num_classes, genotype=kwargs['genotype'])
            else:
                cnn = net(num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, **kwargs)
            self.cnns.append(cnn)
        
        self.mod_aggr = mod_aggr
        self.aggregator = None
        assert(self.mod_aggr in ['MLP', 'avg', 'max', 'none'])
        
        ########## Level in which fuse modalities ##########
        self.feat_dim = self.cnns[0].classifier.in_features  if feat_fusion else self.num_classes
        
        if self.mod_aggr == 'MLP':
            self.aggregator = nn.Sequential(
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.feat_dim * len(self.modalities), self.num_classes)
            )
    
    
    def forward(self, x: Tensor) -> Tensor:
        # print('ConsensusModule3DCNN input shape: {}'.format(x.size()))  # (16, 1, 3, 16, 112, 112)
        cnns_outputs = list()
        cnns_features = list()
        if self.mod_aggr == 'MLP':
            # iterate on the modalities
            for i in range(x.size(1)):
                cnn_output = self.cnns[i](x[:, i, :, :, :, :])
                cnns_outputs.append(cnn_output[0])
                cnns_features.append(cnn_output[1])
            x = torch.cat(cnns_outputs, dim=1)
            x = self.aggregator(x)
        elif self.mod_aggr == 'avg':
            for i in range(x.size(1)):
                cnn_output = self.cnns[i](x[:, i, :, :, :, :])
                cnns_outputs.append(cnn_output[0])
                cnns_features.append(cnn_output[1])
            x = torch.stack(cnns_outputs, dim=1)
            x = x.mean(dim=1)
        elif self.mod_aggr == 'max':
            for i in range(x.size(1)):
                cnn_output = self.cnns[i](x[:, i, :, :, :, :])
                cnns_outputs.append(cnn_output[0])
                cnns_features.append(cnn_output[1])
            x = torch.stack(cnns_outputs, dim=1)
            x = x.max(dim=1)
        elif self.mod_aggr == 'none':  # the case in which use a single modality
            x, cnns_features = self.cnns[0](x[:, 0, :, :, :, :])
            cnns_outputs = x
        else:
            x = None
            cnns_outputs = None
            cnns_features = None
        return x, cnns_outputs, cnns_features

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    
    elif ft_portion == "last_layer":
        for param in model.module.parameters():
            param.requires_grad = False
        for net in model.module.cnns:
            for param in net.classifier.parameters():
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


def get_model(net='resnext', *args, **kwargs):
    if net == 'mobilenetv2': 
        model = ConsensusModule3DCNN(net=mobilenetv2, *args, **kwargs)
    elif net == 'resnext':
        model = ConsensusModule3DCNN(net=resnext, *args, **kwargs)
    elif net == 'tran_renext':
        model = ConsensusModule3DCNN(net=transformer, *args, **kwargs)
    elif net == 'tran_mobilenetv2':
        model = ConsensusModule3DCNN(net=transformer, *args, **kwargs)
    elif net == 'res3d_clstm_mn':
        model = ConsensusModule3DCNN(net=Res3D_cLSTM_MobileNet, *args, **kwargs)
    elif net == 'raar3d':
        from collections import namedtuple
        Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3')
        genotype = Genotype(
            normal1=[('conv_1x1x1', 0), ('conv_3x3x3', 1), ('Max_pool_3x3x3', 0), ('skip_connect', 1), ('conv_1x1x1', 2),
                     ('Max_pool_3x3x3', 0), ('skip_connect', 0), ('conv_3x3x3', 2)], normal_concat1=range(2, 6),
            normal2=[('Max_pool_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_3x3x3', 0), ('skip_connect', 3),
                     ('conv_1x1x1', 2), ('skip_connect', 3), ('skip_connect', 2)], normal_concat2=range(2, 6),
            normal3=[('conv_3x3x3', 1), ('conv_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_1x1x1', 3),
                     ('conv_1x1x1', 2), ('conv_3x3x3', 1), ('conv_1x1x1', 4)], normal_concat3=range(2, 6))
        model = ConsensusModule3DCNN(net=RAAR3DNet.Network, genotype=genotype, *args, **kwargs)
    elif net == 'ni3d':
        from collections import namedtuple
        Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3')
        genotype = Genotype(
            normal1=[('conv_1x1x1', 0), ('conv_3x3x3', 1), ('Max_pool_3x3x3', 0), ('skip_connect', 1), ('conv_1x1x1', 2),
                     ('Max_pool_3x3x3', 0), ('skip_connect', 0), ('conv_3x3x3', 2)], normal_concat1=range(2, 6),
            normal2=[('Max_pool_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_3x3x3', 0), ('skip_connect', 3),
                     ('conv_1x1x1', 2), ('skip_connect', 3), ('skip_connect', 2)], normal_concat2=range(2, 6),
            normal3=[('conv_3x3x3', 1), ('conv_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_1x1x1', 3),
                     ('conv_1x1x1', 2), ('conv_3x3x3', 1), ('conv_1x1x1', 4)], normal_concat3=range(2, 6))
        model = ConsensusModule3DCNN(net=NI3D.Network, genotype=genotype, *args, **kwargs)
    
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