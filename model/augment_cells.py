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


import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from operations import *


def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1. - drop_prob
    mask = 1 - np.random.binomial(
        1, drop_prob, size=[x.shape[0]]).astype(np.float32)
    mask = to_variable(mask)
    x = fluid.layers.elementwise_mul(x / keep_prob, mask, axis=0)
    return x


class AugmentCell(fluid.dygraph.Layer):
    def __init__(self, genotype, c_prev_prev, c_prev, c_curr, reduction,
                 reduction_prev):
        super(AugmentCell, self).__init__()
        print(c_prev_prev, c_prev, c_curr)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        multiplier = len(concat)
        self._multiplier = multiplier
        self._compile(c_curr, op_names, indices, multiplier, reduction)

    def _compile(self, c_curr, op_names, indices, multiplier, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        ops = []
        edge_index = 0
        for op_name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[op_name](c_curr, stride, True)
            ops += [op]
            edge_index += 1
        self._ops = fluid.dygraph.LayerList(ops)
        self._indices = indices

    def forward(self, s0, s1, drop_prob, training):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            states += [h1 + h2]
        out = fluid.layers.concat(input=states[-self._multiplier:], axis=1)
        return out
