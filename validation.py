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

        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs, cnns_outputs, features_outputs = model(inputs)
        # print('Final output: {}\nCnns output: {}\nCNNs features: {}'.format(outputs.size(), cnns_outputs.size(), features_outputs.size()))
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        for ii in range(len(opt.modalities)):
            mod_prec1 = calculate_accuracy(cnns_outputs.data, targets.data, topk=(1,)) if len(opt.modalities)==1 else calculate_accuracy(cnns_outputs[ii].data, targets.data, topk=(1,))
            mods_prec1[opt.modalities[ii]].update(mod_prec1[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        # batch_time.update(time.time() - end_time)
        # end_time = time.time()
        '''
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))
        '''
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