import os
import torch
from torch import nn

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, consensus_module_2dcnn
from models import mobilenetv2, resnext, consensus_module_2dcnn, consensus_module_3dcnn


def generate_model(opt):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet',
                         'shufflenet', 'mobilenetv2', 'shufflenetv2']


    if opt.model == 'c3d':
        from models.c3d import get_fine_tuning_parameters
        model = c3d.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'squeezenet':
        from models.squeezenet import get_fine_tuning_parameters
        model = squeezenet.get_model(
            version=opt.version,
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'shufflenet':
        from models.shufflenet import get_fine_tuning_parameters
        model = shufflenet.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)
    elif opt.model == 'shufflenetv2':
        from models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenet':
        from models.mobilenet import get_fine_tuning_parameters
        model = mobilenet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenetv2':
        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        if opt.model_depth == 50:
            model = resnext.resnext50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnext101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnext152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)



    if opt.gpu is not None:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        '''
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        '''

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            else:
                '''
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5), 
                                nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes))
                '''
                model.module.classifier = nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes)
                #'''
                model.module.classifier = model.module.classifier.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                '''
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.8), 
                                nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes))
                '''
                model.module.classifier = nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes)
                #'''

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()

    
def generate_model_3d(opt):
    assert opt.model in ['resnext', 'mobilenetv2', 'res3d_clstm_mn', 'raar3d', 'ni3d']
    
    from models.consensus_module_3dcnn import get_fine_tuning_parameters
                         
    if opt.model == 'mobilenetv2':
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult,
            net=opt.model,
            modalities=opt.modalities,
            aggr_type=opt.aggr_type)
    elif opt.model == 'resnext':
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            aggr_type=opt.aggr_type,
            feat_fusion=opt.feat_fusion)
    elif opt.model == 'res3d_clstm_mn':
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            aggr_type=opt.aggr_type)
    elif opt.model == 'raar3d':
        # from models.res3d_clstm_mobilenet import get_fine_tuning_parameters
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            aggr_type=opt.aggr_type,
            shallow_layer_num=opt.shallow_layer_num,
            middle_layer_num=opt.middle_layer_num,
            high_layer_num=opt.high_layer_num)
    elif opt.model == 'ni3d':
        # from models.res3d_clstm_mobilenet import get_fine_tuning_parameters
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            aggr_type=opt.aggr_type,
            shallow_layer_num=opt.shallow_layer_num,
            middle_layer_num=opt.middle_layer_num,
            high_layer_num=opt.high_layer_num)

    if opt.gpu is not None:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        '''
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        '''
        
        if opt.pretrain_path:   
            if '.pth' in opt.pretrain_path:
                print('loading pretrained model {}'.format(opt.pretrain_path))
                pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
                assert opt.arch == pretrain['arch']
                if 'jester' in opt.pretrain_path:
                # if len(opt.modalities) == 1:
                    # For model pretrained on Jester
                    state_dict = {key.replace('module.', 'module.cnns.0.'): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.cnns.0.fc', 'module.cnns.0.classifier'): value for key, value in state_dict.items()}
                    
                    # For ex-novo model trained on ChaLearn
                    # state_dict = {key.replace('module.cnns.0.0.', 'module.cnns.0.'): value for key, value in pretrain['state_dict'].items()}
                    # model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(pretrain['state_dict'])
            elif opt.pretrain_path:
                for i in range(len(opt.modalities)):
                    pretrain_path = '_'.join([opt.dataset, opt.model, opt.modalities[i], 'none', 'best.pth'])
                    pretrain_path = os.path.join(opt.pretrain_path, pretrain_path)
                    print('loading pretrained model {}'.format(pretrain_path))
                    pretrain = torch.load(pretrain_path, map_location=torch.device('cpu'))
                    state_dict = {key.replace('module.cnns.0.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.cnns.0.', ''): value for key, value in state_dict.items()}
                    assert opt.arch == pretrain['arch']
                    model.module.cnns[i].load_state_dict(state_dict)
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.n_classes != opt.n_finetune_classes:
                # change the output of the final output
                if opt.aggr_type == 'MLP':
                    model.module.aggregator = nn.Sequential(
                                    # nn.Dropout(0.9),
                                    nn.ReLU(),
                                    nn.Linear(model.module.feat_dim, opt.n_finetune_classes))
                    model.module.aggregator = model.module.aggregator.cuda()
                
                # change the output size of single cnn
                for i in range(len(opt.modalities)):
                    if opt.model == 'mobilenetv2':
                        model.module.cnns[i].classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(model.module.cnns[i].classifier[1].in_features, opt.n_finetune_classes),
                        )
                    elif  opt.model in ['resnext', 'res3d_clstm_mn']:
                        model.module.cnns[i].classifier = nn.Linear(model.module.cnns[i].classifier.in_features, opt.n_finetune_classes)
                    model.module.cnns[i].classifier.cuda()
            
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model in  ['resnext']:
                '''
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.8), 
                                nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes))
                '''
                model.module.classifier = nn.Linear(model.module.classifier.in_features, opt.n_finetune_classes)
                #'''

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()


def generate_model_2d(opt):
    assert opt.model in ['mobilenetv2_2d', 'resnext_2d']


    if opt.model == 'mobilenetv2_2d':        
        from models.consensus_module_2dcnn import get_fine_tuning_parameters
        model = consensus_module_2dcnn.get_model(
            net=opt.model,
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult,
            sample_duration=opt.sample_duration,
            aggr_type=opt.aggr_type)
    elif opt.model == 'resnext_2d':
        assert opt.model_depth in [101]
        from models.consensus_module_2dcnn import get_fine_tuning_parameters
        if opt.model_depth == 101:
            model = consensus_module_2dcnn.get_model(
                net=opt.model,
                arch=opt.model_depth,
                num_classes=opt.n_classes,
                n_finetune_classes=opt.n_finetune_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                aggr_type=opt.aggr_type)

    if opt.gpu is not None:
        model = model.cuda()
        # model = nn.DataParallel(model, device_ids=[0])
        model = nn.DataParallel(model)
        '''
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        '''

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            # change the output of the final output
            if opt.aggr_type == 'MLP':
                model.module.aggregator = nn.Sequential(
                                # nn.Dropout(0.9),
                                nn.ReLU(),
                                nn.Linear(opt.n_finetune_classes * opt.sample_duration, opt.n_finetune_classes))
                model.module.aggregator = model.module.aggregator.cuda()
            elif opt.aggr_type == 'LSTM':
                model.module.aggregator = nn.LSTM(opt.n_finetune_classes, opt.n_finetune_classes, batch_first=False, bidirectional=True)
                model.module.aggregator = model.module.aggregator.cuda()
            '''
                model.module.aggregator = nn.Sequential(
                    nn.LSTM(opt.n_finetune_classes, opt.n_finetune_classes)
            '''
            
            # change the output size of single cnn
            for i in range(opt.sample_duration):
                model.module.cnns[i][0].classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(model.module.cnns[i][0].classifier[1].in_features, opt.n_finetune_classes),
                )
                # print('########## {}° network ##########\n{}################################'.format(i, model.module.cnns[i][0].classifier))
                model.module.cnns[i][0].classifier.cuda()
            # print('########## CNNs ##########\n{}################################'.format(model.module.cnns))
            '''
            else:
                model.module.aggregator = nn.Linear(model.module.aggregator.in_features, opt.n_finetune_classes)
                model.module.aggregator = model.module.aggregator.cuda()
            '''
            
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.aggr_type == 'MLP':
                model.module.aggregator = nn.Sequential(
                                # nn.Dropout(0.9),
                                nn.ReLU(),
                                nn.Linear(model.module.aggregator[1].in_features, opt.n_finetune_classes))
                model.module.aggregator = model.module.aggregator.cuda()
            elif opt.aggr_type == 'LSTM':
                self.aggregator = nn.LSTM(input_size=self.n_finetune_classes, hidden_size=self.n_finetune_classes, batch_first=False, bidirectional=True)
                '''
                self.aggregator = nn.Sequential(
                    nn.LSTM(self.num_classes, self.num_classes),
                    nn.Dropout(0.9),
                    nn.Linear(model.module.aggregator[1].in_features, opt.n_finetune_classes),
                )
                '''
                model.module.aggregator = model.module.aggregator.cuda()
            '''
            else:
                model.module.aggregator = nn.Linear(model.module.aggregator.in_features, opt.n_finetune_classes)
                model.module.aggregator = model.module.aggregator.cuda()
            '''
            
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()