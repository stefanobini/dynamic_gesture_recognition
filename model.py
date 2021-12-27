import os
import torch
from torch import nn

from models import consensus_module_2dcnn, consensus_module_3dcnn, consensus_module_ts

    
def generate_model_3d(opt):
    assert opt.model in ['resnext', 'mobilenetv2', 'res3d_clstm_mn', 'raar3d', 'ni3d', 'EAN_16f']
    
    from models.consensus_module_3dcnn import get_fine_tuning_parameters
                         
    if opt.model == 'mobilenetv2':
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult,
            net=opt.model,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr,
            feat_fusion=opt.feat_fusion,
            ssa_loss=opt.SSA_loss)
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
            mod_aggr=opt.mod_aggr,
            feat_fusion=opt.feat_fusion,
            ssa_loss=opt.SSA_loss)
    elif opt.model == 'res3d_clstm_mn':
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr)
    elif opt.model == 'raar3d':
        # from models.res3d_clstm_mobilenet import get_fine_tuning_parameters
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr,
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
            mod_aggr=opt.mod_aggr,
            shallow_layer_num=opt.shallow_layer_num,
            middle_layer_num=opt.middle_layer_num,
            high_layer_num=opt.high_layer_num)
    elif opt.model == 'EAN_16f':
        # from models.res3d_clstm_mobilenet import get_fine_tuning_parameters
        model = consensus_module_3dcnn.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            net=opt.model,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr
            )

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
                if 'pretrained_models' in opt.pretrain_path:
                # if len(opt.modalities) == 1:
                    # For model pretrained on Jester
                    state_dict = {key.replace('module.', 'module.cnns.0.'): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.cnns.0.fc', 'module.cnns.0.classifier'): value for key, value in state_dict.items()}
                    
                    # For ex-novo model trained on ChaLearn
                    # state_dict = {key.replace('module.cnns.0.0.', 'module.cnns.0.'): value for key, value in pretrain['state_dict'].items()}
                    model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(pretrain['state_dict'])
            else:
                opt.pretrain_path = os.path.join(opt.pretrain_path, opt.dataset, opt.model)
                for i in range(len(opt.modalities)):
                    pretrain_path = '_'.join([opt.dataset, opt.model, opt.modalities[i], 'none', 'best.pth'])
                    pretrain_path = os.path.join(opt.pretrain_path, pretrain_path)
                    print('loading pretrained model {}'.format(pretrain_path))
                    pretrain = torch.load(pretrain_path, map_location=torch.device('cpu'))
                    assert opt.arch == pretrain['arch']
                    # state_dict = {key.replace('module.', ''): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.cnns.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    # state_dict = {key.replace('module.cnns.0.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    model.module.cnns[i].load_state_dict(state_dict)
                    # model.module.cnns[i].load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.n_classes != opt.n_finetune_classes:
                # change the output of the final output
                if opt.mod_aggr == 'MLP':
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
            sample_duration=opt.sample_duration,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr,
            temp_aggr=opt.temp_aggr,
            width_mult=opt.width_mult)
    elif opt.model == 'resnext_2d':
        assert opt.model_depth in [101]
        from models.consensus_module_2dcnn import get_fine_tuning_parameters
        if opt.model_depth == 101:
            model = consensus_module_2dcnn.get_model(
                net=opt.model,
                num_classes=opt.n_classes,
                n_finetune_classes=opt.n_finetune_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                modalities=opt.modalities,
                mod_aggr=opt.mod_aggr,
                temp_aggr=opt.temp_aggr,
                groups=opt.groups,
                width_per_group=opt.resnext_cardinality)

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
                model.load_state_dict(pretrain['state_dict'])
            else:
                opt.pretrain_path = os.path.join(opt.pretrain_path, opt.dataset, opt.model)
                for i in range(len(opt.modalities)):
                    pretrain_path = '_'.join([opt.dataset, opt.model, opt.modalities[i], opt.temp_aggr, 'none', 'best.pth'])
                    pretrain_path = os.path.join(opt.pretrain_path, pretrain_path)
                    print('loading pretrained model {}'.format(pretrain_path))
                    pretrain = torch.load(pretrain_path, map_location=torch.device('cpu'))
                    assert opt.arch == pretrain['arch']
                    # state_dict = {key.replace('module.', ''): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.mod_nets.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    # state_dict = {key.replace('module.cnns.0.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    # print('mode_nets lenght: {}'.format(len(model.module.mod_nets)))
                    model.module.mod_nets[i].load_state_dict(state_dict)
                    # model.module.mod_nets[i].load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            # change the output of the final output
            if opt.temp_aggr == 'MLP':
                model.module.temp_aggregator = nn.Sequential(
                                # nn.Dropout(0.9),
                                nn.ReLU(),
                                nn.Linear(opt.n_finetune_classes * opt.sample_duration, opt.n_finetune_classes))
                model.module.temp_aggregator = model.module.temp_aggregator.cuda()
            elif opt.temp_aggr == 'LSTM':
                model.module.temp_aggregator = nn.LSTM(opt.n_finetune_classes, opt.n_finetune_classes, batch_first=False, bidirectional=True)
                model.module.temp_aggregator = model.module.temp_aggregator.cuda()
            '''
                model.module.aggregator = nn.Sequential(
                    nn.LSTM(opt.n_finetune_classes, opt.n_finetune_classes)
            '''
            if opt.n_classes != opt.n_finetune_classes:
                if opt.mod_aggr == 'MLP':
                    model.module.mod_aggregator = nn.Sequential(
                        # nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(opt.n_finetune_classes * len(opt.modalities), self.n_finetune_classes)
                    )
                    model.module.mod_aggregator = model.module.mod_aggregator.cuda()
                
                for i in range(len(opt.modalities)):
                    # change the output size of single cnn
                    for j in range(opt.sample_duration):
                        model.module.mod_nets[i][j][0].classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(model.module.mod_nets[i][j][0].classifier[1].in_features, opt.n_finetune_classes),
                        )
                        # print('########## {}° network ##########\n{}################################'.format(i, model.module.cnns[i][0].classifier))
                        model.module.mod_nets[i][j][0].classifier.cuda()
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
            
            if opt.temp_aggr == 'MLP':
                model.module.temp_aggregator = nn.Sequential(
                                # nn.Dropout(0.9),
                                nn.ReLU(),
                                nn.Linear(model.module.temp_aggregator[1].in_features, opt.n_finetune_classes))
                model.module.temp_aggregator = model.module.temp_aggregator.cuda()
            elif opt.temp_aggr == 'LSTM':
                self.temp_aggregator = nn.LSTM(input_size=self.n_finetune_classes, hidden_size=self.n_finetune_classes, batch_first=False, bidirectional=True)
                model.module.temp_aggregator = model.module.temp_aggregator.cuda()
            if opt.mod_aggr == 'MLP':
                model.module.mod_aggregator = nn.Sequential(
                    # nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(opt.n_finetune_classes * len(opt.modalities), self.n_finetune_classes)
                )
            
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()


def generate_model_ts(opt):
    assert opt.model in ['timesformer']
    
    from models.consensus_module_ts import get_fine_tuning_parameters
                         
    if opt.model == 'timesformer':
        model = consensus_module_ts.get_model(
            num_classes=opt.n_classes,
            n_finetune_classes=opt.n_finetune_classes,
            sample_size=opt.sample_size,
            net=opt.model,
            modalities=opt.modalities,
            mod_aggr=opt.mod_aggr,
            feat_fusion=opt.feat_fusion,
            ssa_loss=opt.SSA_loss)
    else:
        print('ERROR: Passed model is not compatible.')
        return

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
                model.load_state_dict(pretrain['state_dict'])
            elif opt.pretrain_path:
                opt.pretrain_path = os.path.join(opt.pretrain_path, opt.dataset, opt.model)
                for i in range(len(opt.modalities)):
                    pretrain_path = '_'.join([opt.dataset, opt.model, opt.modalities[i], 'none', 'best.pth'])
                    pretrain_path = os.path.join(opt.pretrain_path, pretrain_path)
                    print('loading pretrained model {}'.format(pretrain_path))
                    pretrain = torch.load(pretrain_path, map_location=torch.device('cpu'))
                    assert opt.arch == pretrain['arch']
                    '''
                    # state_dict = {key.replace('module.', ''): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.nets.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    # state_dict = {key.replace('module.nets.0.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    model.module.nets[i].load_state_dict(state_dict)
                    '''
                    model.module.nets[i].load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.n_classes != opt.n_finetune_classes:
                # change the output of the final output
                if opt.mod_aggr == 'MLP':
                    model.module.aggregator = nn.Sequential(
                                    # nn.Dropout(0.9),
                                    nn.ReLU(),
                                    nn.Linear(model.module.feat_dim, opt.n_finetune_classes))
                    model.module.aggregator = model.module.aggregator.cuda()
                
                # change the output size of single cnn
                for i in range(len(opt.modalities)):
                    model.module.nets[i].model.reset_classifier(opt.n_finetune_classes)
                    model.module.nets[i].model.get_classifier().cuda()
            
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:   
            if '.pth' in opt.pretrain_path:
                print('loading pretrained model {}'.format(opt.pretrain_path))
                pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
                assert opt.arch == pretrain['arch']
                model.load_state_dict(pretrain['state_dict'])
            elif opt.pretrain_path:
                opt.pretrain_path = os.path.join(opt.pretrain_path, opt.dataset, opt.model)
                for i in range(len(opt.modalities)):
                    pretrain_path = '_'.join([opt.dataset, opt.model, opt.modalities[i], 'none', 'best.pth'])
                    pretrain_path = os.path.join(opt.pretrain_path, pretrain_path)
                    print('loading pretrained model {}'.format(pretrain_path))
                    pretrain = torch.load(pretrain_path, map_location=torch.device('cpu'))
                    assert opt.arch == pretrain['arch']
                    '''
                    # state_dict = {key.replace('module.', ''): value for key, value in pretrain['state_dict'].items()}
                    state_dict = {key.replace('module.nets.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    # state_dict = {key.replace('module.nets.0.0.', ''): value for key, value in pretrain['state_dict'].items()}
                    model.module.nets[i].load_state_dict(state_dict)
                    '''
                    model.module.nets[i].load_state_dict(pretrain['state_dict'])
            
            if opt.test or opt.ft_portion == 'none':
                return model, model.parameters()
            
            if opt.n_classes != opt.n_finetune_classes:
                # change the output of the final output
                if opt.mod_aggr == 'MLP':
                    model.module.aggregator = nn.Sequential(
                                    # nn.Dropout(0.9),
                                    nn.ReLU(),
                                    nn.Linear(model.module.feat_dim, opt.n_finetune_classes))
                    model.module.aggregator = model.module.aggregator.cuda()
                
                # change the output size of single cnn
                for i in range(len(opt.modalities)):
                    model.module.nets[i].reset_classifier(opt.n_finetune_classes)
                    # model.module.cnns[i].classifier.cuda()
            
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    
    return model, model.parameters()