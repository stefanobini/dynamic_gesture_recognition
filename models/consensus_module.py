import torch
import torch.nn as nn


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type):
        self.consensus_type = consensus_type
        self.shape = None
    
    @staticmethod
    def forward(self, input_tensor):
        print('Forward input shape: {}'.format(input_tensor.shape))
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(1, keepdim=False)
        elif self.consensus_type == 'max':
            output = input_tensor.max(1, keepdim=False)
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
            grad_in = grad_in.expand(self.shape) / float(self.shape[1])
        else:
            grad_in = None
        
        print('Backward output shape: {}'.format(grad_in.shape))
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, sample_duration, n_class):
        super(ConsensusModule, self).__init__()
        assert(consensus_type in ['avg', 'max'])
        self.consensus_type = consensus_type
        self.consensus = SegmentConsensus(self.consensus_type)
        
        assert(self.aggr_type in ['MLP', 'LSTM', 'avg', 'max'])
        if self.aggr_type == 'MLP':
            self.aggregator = nn.Sequential(
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.num_classes * self.sample_duration, self.num_classes)
            )
        elif self.aggr_type == 'LSTM':
            self.aggregator = nn.Sequential(
                nn.LSTM(self.num_classes, self.num_classes),
                # nn.Dropout(0.2),
                # nn.Linear(self.num_classes * self.sample_duration, self.num_classes),
            )

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
