import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.denoise_arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._denoise_loss(input, target)
        theta = _concat(self.model.denoise_net_parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer'] for v in self.model.denoise_net_parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, self.model.denoise_net_parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._denoise_arch_loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._denoise_arch_loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.denoise_arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.denoise_net_parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.denoise_arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        params, offset = {}, 0
        for v in model_new.denoise_net_parameters():
            v_length = np.prod(v.size())
            v.data.copy_(theta[offset: offset + v_length].view(v.size()))
            offset += v_length
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.denoise_net_parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._denoise_arch_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.denoise_arch_parameters())

        for p, v in zip(self.model.denoise_net_parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._denoise_arch_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.denoise_arch_parameters())

        for p, v in zip(self.model.denoise_net_parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
