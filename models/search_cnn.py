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

import logging
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import genotypes as gt

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NormalInitializer, MSRAInitializer, ConstantInitializer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from genotypes import PRIMITIVES
from operations import *

from models.search_cells import SearchCell


class Network(fluid.dygraph.Layer):
    def __init__(self,
                 args,
                 criterion,
                 aux,
                 alpha_normal=None,
                 alpha_reduce=None,
                 n_nodes=4,
                 stem_multiplier=3):
        super().__init__()
        # NOTE: C_in = args.init_channels here. Origin C_in = input_channels = 3 fixed.
        C_in = args.init_channels
        n_layers = args.layers
        n_classes = args.class_num
        self.n_nodes = n_nodes
        self.epoch=1
        self.aux=aux
        self.criterion = criterion

        if not args.use_gpu:
            place = fluid.CPUPlace()
        elif not args.use_data_parallel:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
        self.place = place

        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = fluid.dygraph.ParameterList()
        self.alpha_reduce = fluid.dygraph.ParameterList()

        if aux:
            for i in range(n_nodes):
                self.alpha_normal.append(
                    paddle.create_parameter(
                        shape=[i+2, n_ops],
                        dtype="float32"))
                self.alpha_reduce.append(
                    paddle.create_parameter(
                        shape=[i+2, n_ops],
                        dtype="float32"))

                with paddle.no_grad():
                    edge_max, primitive_indices = paddle.topk(alpha_normal[i][:, :-1], 1)
                    topk_edge_values, topk_edge_indices = paddle.topk(edge_max.reshape([-1]), 2)
                    for edge_idx in topk_edge_indices:
                        prim_idx = primitive_indices[edge_idx]
                        self.alpha_normal[i][edge_idx][prim_idx] = 1.0
                    
                    edge_max, primitive_indices = paddle.topk(alpha_reduce[i][:, :-1], 1)  # ignore 'none'
                    topk_edge_values, topk_edge_indices = paddle.topk(edge_max.reshape([-1]), 2)
                    for edge_idx in topk_edge_indices:
                        prim_idx = primitive_indices[edge_idx]
                        self.alpha_reduce[i][edge_idx][prim_idx] = 1.0
        else:
            for i in range(n_nodes):
                self.alpha_normal.append(fluid.layers.create_parameter(
                    shape=[i+2, n_ops],
                    dtype="float32",
                    default_initializer=NormalInitializer(
                        loc=0.0, scale=1e-3)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n,p))
        # C_in means the input data
        self.net = SearchCNN(C_in, n_classes, n_layers, "DARTS", steps=n_nodes, stem_multiplier=stem_multiplier)

    def forward(self, x):
        if self.aux:
            weights_normal = [alpha for alpha in self.alpha_normal]
            weights_reduce = [alpha for alpha in self.alpha_reduce]
        else:
            weights_normal = [F.softmax(alpha, axis=0) for alpha in self.alpha_normal]
            weights_reduce = [F.softmax(alpha, axis=0) for alpha in self.alpha_reduce]
        
        with fluid.dygraph.guard(self.place):
            return self.net(x, weights_normal, weights_reduce)
        
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.getLogger('ecdarts').handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, axis=0))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, axis=0))

        # restore formats
        for handler, formatter in zip(logger.getLogger('ecdarts').handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
    def minmaxscaler(data):
        min = np.amin(data)
        max = np.amax(data)    
        return (data - min)/(max-min)


class SearchCNN(fluid.dygraph.Layer):
    def __init__(self,
                 c_in,
                 num_classes,
                 layers,
                 method,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3):
        super(SearchCNN, self).__init__()
        # NOTE: c_in = args.init_channels, input_channels = 3 fixed.
        self._c_in = c_in
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._primitives = PRIMITIVES
        self._method = method

        c_cur = stem_multiplier * c_in
        self.stem = fluid.dygraph.Sequential(
            Conv2D(
                num_channels=3,
                num_filters=c_cur,
                filter_size=3,
                padding=1,
                param_attr=fluid.ParamAttr(initializer=MSRAInitializer()),
                bias_attr=False),
            BatchNorm(
                num_channels=c_cur,
                param_attr=fluid.ParamAttr(
                    initializer=ConstantInitializer(value=1)),
                bias_attr=fluid.ParamAttr(
                    initializer=ConstantInitializer(value=0))))

        c_prev_prev, c_prev, c_cur = c_cur, c_cur, c_in
        cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(steps, multiplier, c_prev_prev, c_prev, c_cur,
                        reduction, reduction_prev, method)
            reduction_prev = reduction
            cells.append(cell)
            c_prev_prev, c_prev = c_prev, multiplier * c_cur
        self.cells = fluid.dygraph.LayerList(cells)
        self.global_pooling = Pool2D(pool_type='avg', global_pooling=True)
        self.classifier = Linear(
            input_dim=c_prev,
            output_dim=num_classes,
            param_attr=ParamAttr(initializer=MSRAInitializer()),
            bias_attr=ParamAttr(initializer=MSRAInitializer()))

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        weights2 = None
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = fluid.layers.softmax(self.alphas_reduce)
                if self._method == "PC-DARTS":
                    n = 3
                    start = 2
                    weights2 = fluid.layers.softmax(self.betas_reduce[0:2])
                    for i in range(self._steps - 1):
                        end = start + n
                        tw2 = fluid.layers.softmax(self.betas_reduce[start:
                                                                     end])
                        start = end
                        n += 1
                        weights2 = fluid.layers.concat([weights2, tw2])
            else:
                weights = fluid.layers.softmax(self.alphas_normal)
                if self._method == "PC-DARTS":
                    n = 3
                    start = 2
                    weights2 = fluid.layers.softmax(self.betas_normal[0:2])
                    for i in range(self._steps - 1):
                        end = start + n
                        tw2 = fluid.layers.softmax(self.betas_normal[start:
                                                                     end])
                        start = end
                        n += 1
                        weights2 = fluid.layers.concat([weights2, tw2])
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        out = fluid.layers.squeeze(out, axes=[2, 3])
        logits = self.classifier(out)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        loss = fluid.layers.reduce_mean(
            fluid.layers.softmax_with_cross_entropy(logits, target))
        return loss

    def new(self):
        model_new = SearchCNN(self._c_in, self._num_classes, self._layers,
                            self._method)
        return model_new

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self._primitives)
        self.alphas_normal = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))
        self.alphas_reduce = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        if self._method == "PC-DARTS":
            self.betas_normal = fluid.layers.create_parameter(
                shape=[k],
                dtype="float32",
                default_initializer=NormalInitializer(
                    loc=0.0, scale=1e-3))
            self.betas_reduce = fluid.layers.create_parameter(
                shape=[k],
                dtype="float32",
                default_initializer=NormalInitializer(
                    loc=0.0, scale=1e-3))
            self._arch_parameters += [self.betas_normal, self.betas_reduce]

    def arch_parameters(self):
        return self._arch_parameters
