# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from architect import Architect

import os
import sys
import ast
import argparse
import functools
import logging
import numpy as np
import utils

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddleslim.nas.darts import architect, count_parameters_in_MB

import reader
from model import IST, clip_grad_norm_
from models.search_cnn import Network
from visualize import plot


sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

parser = argparse.ArgumentParser("Search config")

# yapf: disable
parser.add_argument('--dataset', required=True, help='cifar10/100/tiny_imagenet/imagenet')
parser.add_argument('--log_freq', type=int, default=50, help="Log frequency.")
parser.add_argument('--use_multiprocess', type=bool, default=True, help="Whether use multiprocess reader.")
parser.add_argument('--batch_size', type=int, default=32, help="Minibatch size.")
parser.add_argument('--learning_rate', type=float, default=0.025, help="The start learning rate.")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
parser.add_argument('--use_gpu', type=bool, default=True, help="Whether use GPU.")
parser.add_argument('--init_channels', type=int, default=16, help="Init channel number.")
parser.add_argument('--layers', type=int, default=8, help="Total number of layers.")
parser.add_argument('--n_classes', type=int, default=10, help="Class number of dataset.")
parser.add_argument('--trainset_num', type=int, default=50000, help="images number of trainset.")
parser.add_argument('--model_save_dir', type=str, default='search_cifar', help="The path to save model.")
parser.add_argument('--grad_clip', type=float, default=5, help="Gradient clipping.")
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help="Learning rate for arch encoding.")
parser.add_argument('--method', type=str, default='DARTS', help="The search method you would like to use")
parser.add_argument('--epochs_no_archopt', type=int, default=0, help="Epochs not optimize the arch params")
parser.add_argument('--cutout_length', type=int, default=16, help="Cutout length.")
parser.add_argument('--cutout', type=ast.literal_eval, default=False, help="Whether use cutout.")
parser.add_argument('--unrolled', type=ast.literal_eval, default=False, help="Use one-step unrolled validation loss")
parser.add_argument('--use_data_parallel', type=ast.literal_eval, default=False, help="The flag indicating whether to use data parallel mode to train the model.")

parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
parser.add_argument('--w_weight_decay', type=float, default=3e-4, help='weight decay for weights')
parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--gpus', default='all', help='gpu device ids separated by comma. ''`all` indicates use all gpus.')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='experiment name')
parser.add_argument('--plot_path', type=str, default='./plot/', help='experiment name')
parser.add_argument('--save', type=str, default='EC-DARTS', help='experiment name')
parser.add_argument('--epochs', type=int, default=25, help='# of training epochs')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--workers', type=int, default=16, help='# of workers')
parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
parser.add_argument('--note', type=str, default='try', help='note for this run')
# yapf: enable

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    # NOTE: pass place to IST()
    # NOTE: using paddle.optimizer.Adam()

    # set seed
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    logging.info("args = %s", args)
    
    if args.dataset == 'cifar10':
        train_reader, valid_reader = reader.train_search_cifar10(
            batch_size=args.batch_size,
            train_portion=0.5,
            is_shuffle=True,
            args=args)
    elif args.dataset == 'imagenet':
        args.input_channels = 3
        args.n_classes = 1000
        train_reader = fluid.io.batch(
            reader.imagenet_reader(args.train_dir, 'train'),
            batch_size=args.batch_size,
            drop_last=True)
        valid_reader = fluid.io.batch(
            reader.imagenet_reader(args.val_dir, 'val'),
            batch_size=args.batch_size)
    elif args.dataset == 'tiny-imagenet':
        args.n_classes = 200
        args.input_channels = 3
        raise NotImplementedError

    with fluid.dygraph.guard(place):
        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=False)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=False)

        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(valid_reader, places=place)

        # net_crit = nn.CrossEntropyLoss().to(place)
        # aux_net_crit = nn.CrossEntropyLoss().to(place)
        net_crit = nn.CrossEntropyLoss()
        aux_net_crit = nn.CrossEntropyLoss()
        model = Network(args, net_crit, aux=False)
        # model = model.to(place)

        logging.info("param size = {:.6f}MB".format(
            count_parameters_in_MB(model.parameters())))
        # weights optimizer
        w_optim = paddle.optimizer.Momentum(learning_rate=args.w_lr, momentum=args.w_momentum, parameters=model.weights(), weight_decay=args.w_weight_decay)
        # alphas optimizer
        a_optim = paddle.optimizer.Adam(learning_rate=args.alpha_lr, parameters=model.alphas(), beta1=0.5, beta2=0.999, weight_decay=args.alpha_weight_decay)
        
        # NOTE: CosDecay function in paddle using learning_rate value as input
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.w_lr, T_max=args.epochs, eta_min=args.w_lr_min)
        
        w_optim_aux = paddle.optimizer.Momentum(learning_rate=args.w_lr, momentum=args.w_momentum, parameters=model.weights(), weight_decay=args.w_weight_decay)
        a_optim_aux = paddle.optimizer.Adam(learning_rate=args.alpha_lr, parameters=model.alphas(), beta1=0.5, beta2=0.999, weight_decay=args.alpha_weight_decay)
        lr_scheduler_aux = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.w_lr, T_max=args.epochs, eta_min=args.w_lr_min)

        # architect = Architect(model, v_model, args.w_momentum, args.w_weight_decay)
        architect = Architect(model, args.w_lr, args.alpha_lr, unrolled=args.unrolled, parallel=args.use_data_parallel)

        best_top1 = 0.
        is_best = True
        for epoch in range(args.epochs):
            model.print_alphas(logging)

            # training
            train_acc, tring_obj = train(train_loader, valid_loader, model, architect, w_optim, a_optim, lr_scheduler, epoch)
            logging.info('Train_acc %f', train_acc)

            # validation
            len_train_loader = 0
            for _ in enumerate(train_loader):
                len_train_loader += 1
            cur_step = (epoch+1) * len_train_loader

            model, aux_model, train_aux_acc, train_aux_obj = IST(args, train_loader, valid_loader, model, architect, a_optim_aux, aux_net_crit, w_optim_aux, lr_scheduler_aux, epoch, place, logging)
            logging.info('Train_aux_acc %f', train_aux_acc)

            valid_acc, valid_obj = validate(valid_loader, aux_model, epoch)
            logging.info('Valid_acc %f', valid_acc)
            logging.info('Epoch: %d', epoch)

            # genotype logging
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
            plot_path = os.path.join(args.plot_path, "EP{:02d}".format(epoch+1))
            caption = "Epoch {}".format(epoch+1)
            plot(genotype.normal, plot_path + "-normal", caption)
            plot(genotype.reduce, plot_path + "-reduce", caption)

            # saving
            if best_top1 < valid_acc:
                best_top1 = valid_acc
                best_genotype = genotype
                is_best = True
                save_path = args.save_path + 'aux_model_params.pt'
                paddle.save(aux_model.state_dict(), save_path)
                pretrained_dict = paddle.load(save_path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'alpha' not in k}
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_dict(model_dict)
                # model = model.to(place)
            else:
                is_best = False
            utils.save_checkpoint(model, args.save_path, is_best)
            print("")

            # NOTE: using lr_scheduler after optimizer.step()
            lr_scheduler.step()
        
    logging.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logging.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()

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

        alpha_optim.clear_gradients()
        # architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        architect.step(trn_X, trn_y, val_X, val_y)
        alpha_optim.step()

        w_optim.clear_gradients()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        
        loss.backward()
        clip_grad_norm_(model.weights(), args.w_grad_clip)
        w_optim.step()

        # prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        prec1 = fluid.layers.accuracy(input=logits, label=trn_y, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=trn_y, k=5)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        # if step % args.print_freq == 0 or step == len(train_loader) - 1:
        if step % args.print_freq == 0:
            logging.info('Aux_TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)
        
    return top1.avg, losses.avg


def validate(valid_loader, model, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with paddle.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X = to_variable(X)
            y = to_variable(y)
            N = X.shape[0]

            logits = model(X)
            loss = model.criterion(logits, y)

            # prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            prec1 = fluid.layers.accuracy(input=logits, label=y, k=1)
            prec5 = fluid.layers.accuracy(input=logits, label=y, k=5)
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % args.print_freq == 0:
                logging.info('Epoch: %d VALID Step: %03d Objs: %e R1: %f R5: %f', epoch, step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
