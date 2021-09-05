import torch
import torch.nn as nn
from operations import *
from models.search_cnn import Network
import utils

import paddle
# import paddle.nn as nn


def IST(args, train_loader, valid_loader, model, architect, alpha_optim, aux_net_crit, aux_w_optim, lr_scheduler_aux, epoch, device,
        logging):
    lr_scheduler_aux.step()
    lr = lr_scheduler_aux.get_lr()[0]
    save_path = args.save_path + 'one-shot_weights.pt'
    torch.save(model.state_dict(), save_path)
    pretrained_dict = torch.load(save_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'alpha' not in k}

    # construct an auxiliary model
    aux_model = Network(args, aux_net_crit, aux=True, alpha_normal=model.alpha_normal, alpha_reduce=model.alpha_reduce,
                            device_ids=args.gpus)
    aux_model_dict = aux_model.state_dict()
    aux_model_dict.update(pretrained_dict)
    aux_model.load_state_dict(aux_model_dict)
    aux_model = aux_model.to(device)
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)

    aux_model.train()
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # # # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, aux_w_optim)
        alpha_optim.step()

        aux_w_optim.zero_grad()
        logits = aux_model(trn_X)

        loss = aux_model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(aux_model.weights(), args.w_grad_clip)
        aux_w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info('Aux_TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)
        cur_step += 1

    return model, aux_model, top1.avg, losses.avg
