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

import paddle.fluid as fluid
from paddle.fluid.layers.nn import shape
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from genotypes import PRIMITIVES
from operations import *


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = fluid.layers.reshape(
        x, [batchsize, groups, channels_per_group, height, width])
    x = fluid.layers.transpose(x, [0, 2, 1, 3, 4])

    # flatten
    x = fluid.layers.reshape(x, [batchsize, num_channels, height, width])
    return x


class MixedOp(fluid.dygraph.Layer):
    def __init__(self, c_cur, stride, method):
        super(MixedOp, self).__init__()
        self._method = method
        self._k = 4 if self._method == "PC-DARTS" else 1
        self.mp = Pool2D(
            pool_size=2,
            pool_stride=2,
            pool_type='max', )
        ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](c_cur // self._k, stride, False)
            if 'pool' in primitive:
                gama = ParamAttr(
                    initializer=fluid.initializer.Constant(value=1),
                    trainable=False)
                beta = ParamAttr(
                    initializer=fluid.initializer.Constant(value=0),
                    trainable=False)
                BN = BatchNorm(
                    c_cur // self._k, param_attr=gama, bias_attr=beta)
                op = fluid.dygraph.Sequential(op, BN)
            ops.append(op)
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, x, weights):
        out = fluid.layers.sums(
            [weights[i] * op(x) for i, op in enumerate(self._ops)])
        return out


class SearchCell(fluid.dygraph.Layer):
    def __init__(self, steps, multiplier, c_prev_prev, c_prev, c_cur,
                 reduction, reduction_prev, method):
        super(SearchCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_cur, False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_cur, 1, 1, 0, False)
        self.preprocess1 = ReLUConvBN(c_prev, c_cur, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._method = method

        ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(c_cur, stride, method)
                ops.append(op)
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, s0, s1, weights, weights2=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = fluid.layers.sums([
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            ])
            offset += len(states)
            states.append(s)
        out = fluid.layers.concat(input=states[-self._multiplier:], axis=1)
        return out
