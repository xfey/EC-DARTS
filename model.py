from operations import *
from models.search_cnn import Network
import utils

import paddle
import paddle.fluid as fluid
# import paddle.nn as nn
from paddle.fluid.dygraph.base import to_variable


def IST(args, train_loader, valid_loader, model, architect, alpha_optim, aux_net_crit, aux_w_optim, lr_scheduler_aux, epoch, place,
        logging):
    save_path = args.save_path + 'one-shot_weights.pt'
    paddle.save(model.state_dict(), save_path)
    pretrained_dict = paddle.load(save_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'alpha' not in k}

    # construct an auxiliary model
    aux_model = Network(args, aux_net_crit, aux=True, alpha_normal=model.alpha_normal, alpha_reduce=model.alpha_reduce)
    aux_model_dict = aux_model.state_dict()
    aux_model_dict.update(pretrained_dict)
    aux_model.load_state_dict(aux_model_dict)
    
    aux_model = aux_model.to(place)
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    len_train_loader = 0
    for _ in enumerate(train_loader):
        len_train_loader += 1

    cur_step = epoch * len_train_loader

    aux_model.train()
    # for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
    #     trn_X, trn_y = trn_X.to(place, blocking=False), trn_y.to(place, blocking=False)
    #     val_X, val_y = val_X.to(place, blocking=False), val_y.to(place, blocking=False)
    #     N = trn_X.size(0)
    for step, (train_data, valid_data) in enumerate(zip(train_loader(), valid_loader())):
        trn_X, trn_y = train_data
        val_X, val_y = valid_data
        trn_X = to_variable(trn_X)
        trn_y = to_variable(trn_y)
        trn_y.stop_gradient = True
        val_X = to_variable(val_X)
        val_y = to_variable(val_y)
        val_y.stop_gradient = True
        N = trn_X.shape[0]

        # # # phase 2. architect step (alpha)
        alpha_optim.clear_gradients()
        # architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, aux_w_optim)
        architect.step(trn_X, trn_y, val_X, val_y)
        alpha_optim.step()

        aux_w_optim.clear_gradients()
        logits = aux_model(trn_X)

        loss = aux_model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        clip_grad_norm_(aux_model.weights(), args.w_grad_clip)
        aux_w_optim.step()

        # prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        prec1 = fluid.layers.accuracy(input=logits, label=trn_y, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=trn_y, k=5)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        # if step % args.print_freq == 0 or step == len(train_loader) - 1:
        if step % args.print_freq == 0:
            logging.info('Aux_TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)
        cur_step += 1

    lr_scheduler_aux.step()
    
    return model, aux_model, top1.avg, losses.avg


def clip_grad_norm_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)
