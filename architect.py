""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
# import torch
from numpy.linalg import eigvals
import numpy as np


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, v_net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        # self.v_net = copy.deepcopy(net)
        self.v_net = v_net
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.hessian = None

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        # gradients = torch.autograd.grad(loss, self.net.weights())
        paddle.enable_static()
        gradients = paddle.fluid.backward.gradients(loss, self.net.weights())
        paddle.disable_static()

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        # with torch.no_grad():
        with paddle.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        
        # v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        paddle.enable_static()
        v_grads = paddle.fluid.backward.gradients(loss, v_alphas + v_weights)
        paddle.disable_static()

        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        # with torch.no_grad():
        with paddle.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
