import torch
from torch.autograd import Variable
import time
import sys
from tqdm import tqdm

from utils import *


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    # print('validation at epoch {}'.format(epoch))

    model.eval()

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mods_prec1 = dict()
    for modality in opt.modalities:
        mods_prec1[modality] = AverageMeter()
        

    end_time = time.time()
    batch_iter = tqdm(enumerate(data_loader), 'Validation at epoch {:03d}'.format(epoch), total=len(data_loader))
    for i, (inputs, targets) in batch_iter:
        # data_time.update(time.time() - end_time)

        if opt.gpu is not None:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        if opt.cnn_dim == 3:
            # outputs = model(inputs)
            outputs, cnns_outputs, features_outputs = model(inputs)
            # outputs, features_outputs = model(inputs)
        elif opt.cnn_dim == 2:
            outputs, cnns_outputs = model(inputs)
        else:
            print('ERROR: "cnn_dim={}" is not acceptable.'.format(opt.cnn_dim))
        '''
        print('************** VALIDATION **************\n')
        print('Final output: {}\nCnns output: {}\nCNNs features: {}'.format(outputs.size(), cnns_outputs.size(), features_outputs.size() if (features_outputs is not None) else features_outputs))
        print('****************************************\n')
        '''
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        for ii in range(len(opt.modalities)):
            mod_prec1 = calculate_accuracy(cnns_outputs.data, targets.data, topk=(1,)) if len(opt.modalities)==1 else calculate_accuracy(cnns_outputs[ii].data, targets.data, topk=(1,))
            # mod_prec1 = calculate_accuracy(outputs.data, targets.data, topk=(1,)) if len(opt.modalities)==1 else calculate_accuracy(cnns_outputs[ii].data, targets.data, topk=(1,))
            mods_prec1[opt.modalities[ii]].update(mod_prec1[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        # batch_time.update(time.time() - end_time)
        # end_time = time.time()
        
        # batch_iter.set_description(f'Validation at epoch {epoch:03d}')  # update progressbar
        # batch_iter.set_description(f'Validation at epoch {epoch:03d}, avgLoss: {losses.avg.item():.4f}, avgPrec@1: {top1.avg.item():.2f}, avgPrec@5: {top5.avg.item():.2f}')  # update progressbar
    batch_iter.close()
    
    log_dict = {'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()}
    mods_prec1_list = list()
    for modality, prec1 in mods_prec1.items():
        log_dict[modality+'_prec1'] = prec1.avg.item()
        mods_prec1_list.append(prec1.avg.item())
    logger.log(log_dict)
    

    return losses.avg.item(), top1.avg.item(), mods_prec1_list