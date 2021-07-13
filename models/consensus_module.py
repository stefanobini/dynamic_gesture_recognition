import torch
import torch.nn as nn


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=0):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None
    
    @staticmethod
    def forward(self, input_tensor):
        print('Forward input shape: {}'.format(input_tensor.shape))
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'max':
            output = input_tensor.max(dim=self.dim, keepdim=True)
        else:
            output = None
        self.save_for_backward(output)
        print('Forward output shape: {}'.format(output.shape))
        return output
    
    @staticmethod
    def backward(self, grad_output):
        print('Backward input shape: {}'.format(grad_output.shape))
        grad_in = self.saved_tensors
        
        if self.consensus_type in ['avg', 'max']:
            grad_in = grad_in.expand(self.shape) / float(self.shape[self.dim])
        else:
            grad_in = None
        
        print('Backward output shape: {}'.format(grad_in.shape))
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=0):
        super(ConsensusModule, self).__init__()
        assert(consensus_type in ['avg', 'max'])
        self.consensus_type = consensus_type
        self.dim = dim
        self.consensus = SegmentConsensus(self.consensus_type, self.dim)

    def forward(self, input):
        # input = torch.stack(input)
        print('Consensus stack size: {}'.format(input.shape))
        return self.consensus(input)


class MLPmodule(torch.nn.Module):
    """
    This is the 1-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """
    def __init__(self, num_frames, num_class):
        super(MLPmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Linear((self.num_class * self.num_frames), self.num_class)
        )
        '''
        self.num_bottleneck = 512
        self.classifier = nn.Sequential(
                                       nn.ReLU(),
                                       nn.Linear(self.num_frames * self.num_class, self.num_bottleneck))
                                       # nn.Dropout(0.90), # Add an extra DO if necess.
                                       nn.ReLU(),
                                       nn.Linear(self.num_bottleneck, self.num_class),
        )
        '''
        
    def forward(self, input):
        # input = torch.cat(input)
        input = self.classifier(input)
        return input


def return_MLP(relation_type,num_frames, num_class):
    MLPmodel = MLPmodule(num_frames, num_class)

    return MLPmodel
