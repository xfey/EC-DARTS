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

""" training for architecture in cifar10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import ast
import logging
import argparse
import functools
import numpy as np
from collections import namedtuple

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid.dygraph.base import to_variable
from paddleslim.common import AvgrageMeter, get_logger
from paddleslim.nas.darts import count_parameters_in_MB

import genotypes
import reader
from models.augment_cnn import NetworkCIFAR as Network
from genotypes import EC_DARTS

sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser("Training Config")

# yapf: disable
parser.add_argument('--use_multiprocess', type=bool, default=True, help="Whether use multiprocess reader.")
parser.add_argument('--data', type=str, default='dataset/cifar10', help="The dir of dataset.")
parser.add_argument('--batch_size', type=int, default=64, help="Minibatch size.")
parser.add_argument('--learning_rate', type=float, default=0.025, help="The start learning rate.")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
parser.add_argument('--weight_decay', type=float, default=3e-4, help="Weight_decay.")
parser.add_argument('--use_gpu', type=bool, default=True, help="Whether use GPU.")
parser.add_argument('--layers', type=int, default=20, help="Total number of layers.")
parser.add_argument('--n_classes', type=int, default=10, help="Class number of dataset.")
parser.add_argument('--trainset_num', type=int, default=50000, help="images number of trainset.")
parser.add_argument('--model_save_dir', type=str, default='eval_cifar', help="The path to save model.")
parser.add_argument('--cutout', type=bool, default=True, help='Whether use cutout.')
parser.add_argument('--auxiliary', type=bool, default=True, help='Use auxiliary tower.')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help="Weight for auxiliary loss.")
parser.add_argument('--log_freq', type=int, default=50, help='Report frequency')
parser.add_argument('--use_data_parallel', type=ast.literal_eval, default=False, help="The flag indicating whether to use data parallel mode to train the model.")

parser.add_argument('--dataset', required=True, help='cifar10/100/tiny_imagenet/imagenet')
parser.add_argument('--train_dir', type=str, default='/home/datasets/', help='')
parser.add_argument('--val_dir', type=str, default='/home/datasets/', help='')
parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping for weights')
parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
parser.add_argument('--gpus', default='all', help='gpu device ids separated by comma. ' '`all` indicates use all gpus.')
parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
parser.add_argument('--init_channels', type=int, default=34)
parser.add_argument('--n_layers', type=int, default=20, help='# of layers')
parser.add_argument('--arch', type=str, default="EC_DARTS", help='which architecture to use')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--workers', type=int, default=16, help='# of workers')
parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
parser.add_argument('--save', type=str, default='./retrain', help='experiment name')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=8)
parser.add_argument('--note', type=str, default='try', help='note for this run')
# yapf: enable

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    
    # set seed
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    logging.info("args = %s", args)

    if args.arch== 'EC_DARTS':
        genotype = EC_DARTS
        print('---------Genotype---------')
        logging.info(genotype)
        print('--------------------------')
    else:
        print('Architect error')
        exit(1)
    
    if args.dataset != 'cifar10':
        print('This code is for cifar10 only. Train in imagenet by train_imagenet.py')
        exit(1)

    with fluid.dygraph.guard(place):
        criterion = nn.CrossEntropyLoss().to(place)
        use_aux = args.aux_weight > 0.

        model = Network(
            C=args.init_channels,
            num_classes=args.n_classes,
            layers=args.layers,
            auxiliary=args.auxiliary,
            genotype=genotype)

        logger.info("param size = {:.6f}MB".format(
            count_parameters_in_MB(model.parameters())))

        device_num = fluid.dygraph.parallel.Env().nranks
        step_per_epoch = int(args.trainset_num / (args.batch_size * device_num))
        learning_rate = fluid.dygraph.CosineDecay(args.learning_rate,
                                                  step_per_epoch, args.epochs)
        clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=args.grad_clip)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            momentum=args.momentum,
            regularization=fluid.regularizer.L2Decay(args.weight_decay),
            parameter_list=model.parameters(),
            grad_clip=clip)

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=args.use_multiprocess)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=args.use_multiprocess)

        train_reader = reader.train_valid(
            batch_size=args.batch_size,
            is_train=True,
            is_shuffle=True,
            args=args)
        valid_reader = reader.train_valid(
            batch_size=args.batch_size,
            is_train=False,
            is_shuffle=False,
            args=args)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(valid_reader, places=place)

        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        best_acc = 0
        for epoch in range(args.epochs):
            drop_path_prob = args.drop_path_prob * epoch / args.epochs
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch, optimizer.current_step_lr()))
            train_top1 = train(model, train_loader, optimizer, epoch,
                               drop_path_prob, args)
            logger.info("Epoch {}, train_acc {:.6f}".format(epoch, train_top1))
            valid_top1 = valid(model, valid_loader, epoch, args)
            if valid_top1 > best_acc:
                best_acc = valid_top1
                if save_parameters:
                    fluid.save_dygraph(model.state_dict(),
                                       args.model_save_dir + "/best_model")
            logger.info("Epoch {}, valid_acc {:.6f}, best_valid_acc {:.6f}".
                        format(epoch, valid_top1, best_acc))


def train(model, train_reader, optimizer, epoch, drop_path_prob, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step_id, data in enumerate(train_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        label.stop_gradient = True
        logits, logits_aux = model(image, drop_path_prob, True)

        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            fluid.layers.softmax_with_cross_entropy(logits, label))
        if args.auxiliary:
            loss_aux = fluid.layers.reduce_mean(
                fluid.layers.softmax_with_cross_entropy(logits_aux, label))
            loss = loss + args.auxiliary_weight * loss_aux

        if args.use_data_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()

        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)

        if step_id % args.log_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]


def valid(model, valid_reader, epoch, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for step_id, data in enumerate(valid_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        logits, _ = model(image, 0, False)
        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            fluid.layers.softmax_with_cross_entropy(logits, label))

        n = image.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)
        if step_id % args.log_freq == 0:
            logger.info(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
