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
import argparse
import functools

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import reader
from model_search import Network
from paddleslim.nas.darts import DARTSearch
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

parser = argparse.ArgumentParser("Search config")

# yapf: disable
parser.add_argument('--log_freq', type=int, default=50, help="Log frequency.")
parser.add_argument('--use_multiprocess', type=bool, default=True, help="Whether use multiprocess reader.")
parser.add_argument('--batch_size', type=int, default=32, help="Minibatch size.")
parser.add_argument('--learning_rate', type=float, default=0.025, help="The start learning rate.")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
parser.add_argument('--use_gpu', type=bool, default=True, help="Whether use GPU.")
parser.add_argument('--epochs', type=int, default=50, help="Epoch number.")
parser.add_argument('--init_channels', type=int, default=16, help="Init channel number.")
parser.add_argument('--layers', type=int, default=8, help="Total number of layers.")
parser.add_argument('--class_num', type=int, default=10, help="Class number of dataset.")
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
# yapf: enable

args = parser.parse_args()

def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    train_reader, valid_reader = reader.train_search(
        batch_size=args.batch_size,
        train_portion=0.5,
        is_shuffle=True,
        args=args)

    with fluid.dygraph.guard(place):
        model = Network(args.init_channels, args.class_num, args.layers,
                        args.method)
        searcher = DARTSearch(
            model,
            train_reader,
            valid_reader,
            place,
            learning_rate=args.learning_rate,
            batchsize=args.batch_size,
            num_imgs=args.trainset_num,
            arch_learning_rate=args.arch_learning_rate,
            unrolled=args.unrolled,
            num_epochs=args.epochs,
            epochs_no_archopt=args.epochs_no_archopt,
            use_multiprocess=args.use_multiprocess,
            use_data_parallel=args.use_data_parallel,
            save_dir=args.model_save_dir,
            log_freq=args.log_freq)
        searcher.train()


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
