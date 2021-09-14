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
from models.augment_cnn import NetworkImageNet as Network
from genotypes import EC_DARTS_image as EC_DARTS


sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser("Training Imagenet Config")

# yapf: disable
parser.add_argument('--use_multiprocess',  type=bool,  default=True,            help="Whether use multiprocess reader.")
parser.add_argument('--num_workers',       type=int,   default=4,               help="The multiprocess reader number.")
parser.add_argument('--batch_size',        type=int,   default=32,              help="Minibatch size.")
parser.add_argument('--learning_rate',     type=float, default=0.1,             help="The start learning rate.")
parser.add_argument('--decay_rate',        type=float, default=0.97,            help="The lr decay rate.")
parser.add_argument('--momentum',          type=float, default=0.9,             help="Momentum.")
parser.add_argument('--weight_decay',      type=float, default=3e-5,            help="Weight_decay.")
parser.add_argument('--use_gpu',           type=bool,  default=True,            help="Whether use GPU.")
parser.add_argument('--layers',            type=int,   default=14,              help="Total number of layers.")
parser.add_argument('--n_classes',         type=int,   default=1000,            help="Class number of dataset.")
parser.add_argument('--trainset_num',      type=int,   default=1281167,         help="Images number of trainset.")
parser.add_argument('--model_save_dir',    type=str,   default='eval_imagenet', help="The path to save model.")
parser.add_argument('--auxiliary',         type=bool,  default=True,            help='Use auxiliary tower.')
parser.add_argument('--auxiliary_weight',  type=float, default=0.4,             help="Weight for auxiliary loss.")
parser.add_argument('--dropout',           type=float, default=0.0,             help="Dropout probability.")
parser.add_argument('--label_smooth',      type=float, default=0.1,             help="Label smoothing.")
parser.add_argument('--log_freq',          type=int,   default=100,             help='Report frequency')
parser.add_argument('--use_data_parallel', type=ast.literal_eval,  default=False, help="The flag indicating whether to use data parallel mode to train the model.")

parser.add_argument('--data_dir',          type=str,   default='/hdd1/hdd_A/imagenet/data/', help="The dir of dataset.")
parser.add_argument('--dataset', required=True, help='cifar10/100/tiny_imagenet/imagenet')
parser.add_argument('--train_dir', type=str, default='/hdd1/hdd_A/imagenet/data/ILSVRC2012_img_train', help='')
parser.add_argument('--val_dir', type=str, default='/hdd1/hdd_A/imagenet/data/ILSVRC2012_img_val', help='')
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
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)

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
    
    # if args.dataset != 'imagenet':
    #     print('This code is for imagenet only. Train in cifar -> train.py')
    #     exit(1)
    
    with fluid.dygraph.guard(place):
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(
            C=args.init_channels,
            num_classes=args.n_classes,
            layers=args.layers,
            auxiliary=args.auxiliary,
            genotype=genotype)

        logger.info("param size = {:.6f}MB".format(
            count_parameters_in_MB(model.parameters())))

        device_num = fluid.dygraph.parallel.Env().nranks
        step_per_epoch = int(args.trainset_num /
                             (args.batch_size * device_num))
        learning_rate = fluid.dygraph.ExponentialDecay(
            args.learning_rate,
            step_per_epoch,
            args.decay_rate,
            staircase=True)

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
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)

        train_reader = fluid.io.batch(
            reader.imagenet_reader(args.train_dir, 'train'),
            batch_size=args.batch_size,
            drop_last=True)
        valid_reader = fluid.io.batch(
            reader.imagenet_reader(args.val_dir, 'val'),
            batch_size=args.batch_size)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        train_loader.set_sample_list_generator(train_reader, places=place)
        valid_loader.set_sample_list_generator(valid_reader, places=place)

        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        best_top1 = 0
        for epoch in range(args.epochs):
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch, optimizer.current_step_lr()))
            train_top1, train_top5 = train(model, train_loader, optimizer,
                                           epoch, args)
            logger.info("Epoch {}, train_top1 {:.6f}, train_top5 {:.6f}".
                        format(epoch, train_top1, train_top5))
            valid_top1, valid_top5 = valid(model, valid_loader, epoch, args)
            if valid_top1 > best_top1:
                best_top1 = valid_top1
                if save_parameters:
                    fluid.save_dygraph(model.state_dict(),
                                       args.model_save_dir + "/best_model")
            logger.info(
                "Epoch {}, valid_top1 {:.6f}, valid_top5 {:.6f}, best_valid_top1 {:6f}".
                format(epoch, valid_top1, valid_top5, best_top1))


def cross_entropy_label_smooth(preds, targets, epsilon):
    preds = fluid.layers.softmax(preds)
    targets_one_hot = fluid.one_hot(input=targets, depth=args.n_classes)
    targets_smooth = fluid.layers.label_smooth(
        targets_one_hot, epsilon=epsilon, dtype="float32")
    targets_smooth_rs = paddle.reshape(targets_smooth, [128, 1000])
    loss = fluid.layers.cross_entropy(
        input=preds, label=targets_smooth_rs, soft_label=True)
    return loss


def train(model, train_reader, optimizer, epoch, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step_id, data in enumerate(train_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        label.stop_gradient = True
        logits, logits_aux = model(image, True)

        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            cross_entropy_label_smooth(logits, label, args.label_smooth))

        if args.auxiliary:
            loss_aux = fluid.layers.reduce_mean(
                cross_entropy_label_smooth(logits_aux, label,
                                           args.label_smooth))
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
    return top1.avg[0], top5.avg[0]


def valid(model, valid_reader, epoch, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for step_id, data in enumerate(valid_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        logits, _ = model(image, False)
        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            cross_entropy_label_smooth(logits, label, args.label_smooth))

        n = image.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)
        if step_id % args.log_freq == 0:
            logger.info(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0], top5.avg[0]


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
