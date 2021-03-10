from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import itertools
import numpy as np
import genotypes
from collections import OrderedDict


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SearchBlock(nn.Module):

    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()

        self.stride = 1
        self.channel = channel

        op_names, indices = zip(*genotype.normal)

        self.dc = self.distilled_channels = self.channel  # // 2
        self.rc = self.remaining_channels = self.channel
        self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.c2_d = OPS[op_names[2]](self.channel, self.dc)
        self.c2_r = OPS[op_names[3]](self.channel, self.rc)
        self.c3_d = OPS[op_names[4]](self.channel, self.dc)
        self.c3_r = OPS[op_names[5]](self.channel, self.rc)
        self.c4 = OPS[op_names[6]](self.channel, self.dc)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.c5 = conv_layer(self.dc * 4, self.channel, 1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        # out_fused = self.esa(self.c5(out))
        out_fused = self.c5(out)

        return out_fused


class IEM(nn.Module):
    def __init__(self, channel, genetype):
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype

        self.cell = SearchBlock(self.channel, self.genetype)
        self.activate = nn.Sigmoid()

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input_y, input_u, k):
        if k == 0:
            t_hat = self.max_operation(input_y)
        else:
            t_hat = self.max_operation(input_u) - 0.5 * (input_u - input_y)
        t = t_hat
        t = self.cell(t)
        t = self.activate(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(input_y / t, 0.0, 1.0)

        return u, t


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel, genotype):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin
        self.channel = channel
        self.genotype = genotype

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(self.channel, self.genotype))

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input):
        t_list = []
        u_list = []
        u = torch.ones_like(input)
        for i in range(self.iem_nums):
            u, t = self.iems[i](input, u, i)
            u_list.append(u)
            t_list.append(t)
        return u_list, t_list


class DenoiseNetwork(nn.Module):

    def __init__(self, layers, channel, genotype):
        super(DenoiseNetwork, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        self.genotype = genotype
        self.stem = conv_layer(3, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel, genotype))
        self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input):

        feat = self.stem(input)
        for i in range(self.nrm_nums):
            feat = self.nrms[i](feat)
        n = self.activate(feat)
        output = input - n
        return output, n


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.iem_nums = 3
        self.nrm_nums = 3
        self.enhance_channel = 3
        self.denoise_channel = 6

        self._criterion = LossFunction()
        self._denoise_criterion = DenoiseLossFunction()

        enhance_genname = 'IEM'
        enhance_genotype = eval("genotypes.%s" % enhance_genname)

        denoise_genname = 'NRM'
        denoise_genotype = eval("genotypes.%s" % denoise_genname)

        self.enhance_net = EnhanceNetwork(iteratioin=self.iem_nums, channel=self.enhance_channel,
                                          genotype=enhance_genotype)
        self.denoise_net = DenoiseNetwork(layers=self.nrm_nums, channel=self.denoise_channel, genotype=denoise_genotype)

        self.enhancement_optimizer = torch.optim.SGD(
            self.enhance_net.parameters(),
            lr=0.015,
            momentum=0.9,
            weight_decay=3e-4)

        self.denoise_optimizer = torch.optim.SGD(
            self.denoise_net.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=3e-4)

        self._init_weights()

    def _init_weights(self):
        model_dict = torch.load('./model/denoise.pt')
        self.denoise_net.load_state_dict(model_dict)

    def forward(self, input):
        u_list, t_list = self.enhance_net(input)
        u_d, noise = self.denoise_net(u_list[-1])
        u_list.append(u_d)
        return u_list, t_list

    def _loss(self, input, target):
        u_list, t_listt = self(input)
        enhance_loss = self._criterion(input, u_list, t_listt)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return enhance_loss + denoise_loss

    def _enhcence_loss(self, input, target):
        u_list, t_listt = self(input)
        enhance_loss = self._criterion(input, u_list, t_listt)
        return enhance_loss

    def _denoise_loss(self, input, target):
        u_list, t_listt = self(input)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return denoise_loss

    def optimizer(self, input, target, step):

        u_list, t_listt = self(input)
        self.enhancement_optimizer.zero_grad()
        enhancement_loss = self._criterion(input, u_list, t_listt)
        enhancement_loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm(self.enhance_net.parameters(), 5)
        self.enhancement_optimizer.step()

        denoise_loss = 0
        if step % 50 == 0:
            self.denoise_optimizer.zero_grad()
            denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
            denoise_loss.backward()
            nn.utils.clip_grad_norm(self.denoise_net.parameters(), 5)
            self.denoise_optimizer.step()

        return enhancement_loss, denoise_loss, u_list


class DenoiseLossFunction(nn.Module):
    def __init__(self):
        super(DenoiseLossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        return 0.0000001 * self.l2_loss(output, target) + self.tv_loss(output)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, u_list, t_list):
        Fidelity_Loss = 0
        # Fidelity_Loss = Fidelity_Loss + self.l2_loss(output_list[i], input_list[i])
        # Smooth_Loss = 0
        # Smooth_Loss = Smooth_Loss + self.smooth_loss(input_list[i], output_list[i])
        i = input
        o = t_list[-1]
        # for i, o in zip(input_list, output_list):
        Fidelity_Loss = Fidelity_Loss + self.l2_loss(o, i)

        Smooth_Loss = 0
        # for i, o in zip(input_list, output_list):
        Smooth_Loss = Smooth_Loss + self.smooth_loss(i, o)
        # for d in d_list[:-1]:
        #   Smooth_Loss = Smooth_Loss + self.smooth_loss(input_list[0], d)
        # Alpha_Loss = 0
        # for a in alpha_list:
        #     Alpha_Loss = Alpha_Loss + self.smooth_loss(input_list[0], a)

        return 0.5 * Fidelity_Loss + Smooth_Loss


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 0.1

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(1, 3, input_im.shape[2], input_im.shape[3])
        return out

    def norm(self, tensor, p):
        return torch.mean(torch.pow(torch.abs(tensor), p))

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        # print(self.input.shape)
        sigma_color = -1.0 / 2 * self.sigma * self.sigma
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * self.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p)
        pixel_grad2 = w2 * self.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p)
        pixel_grad3 = w3 * self.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p)
        pixel_grad4 = w4 * self.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p)
        pixel_grad5 = w5 * self.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p)
        pixel_grad6 = w6 * self.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p)
        pixel_grad7 = w7 * self.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p)
        pixel_grad8 = w8 * self.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p)
        pixel_grad9 = w9 * self.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p)
        pixel_grad10 = w10 * self.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p)
        pixel_grad11 = w11 * self.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p)
        pixel_grad12 = w12 * self.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p)
        pixel_grad13 = w13 * self.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p)
        pixel_grad14 = w14 * self.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p)
        pixel_grad15 = w15 * self.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p)
        pixel_grad16 = w16 * self.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p)
        pixel_grad17 = w17 * self.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p)
        pixel_grad18 = w18 * self.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p)
        pixel_grad19 = w19 * self.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p)
        pixel_grad20 = w20 * self.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p)
        pixel_grad21 = w21 * self.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p)
        pixel_grad22 = w22 * self.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p)
        pixel_grad23 = w23 * self.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p)
        pixel_grad24 = w24 * self.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term


class IlluLoss(nn.Module):
    def __init__(self):
        super(IlluLoss, self).__init__()

    def forward(self, input_I_low, input_im):
        input_gray = self.rgb_to_gray(input_im)
        low_gradient_x, low_gradient_y = self.compute_image_gradient(input_I_low)
        input_gradient_x, input_gradient_y = self.compute_image_gradient(input_gray)

        less_location_x = input_gradient_x < 0.01
        input_gradient_x = input_gradient_x.masked_fill_(less_location_x, 0.01)
        less_location_y = input_gradient_y < 0.01
        input_gradient_y = input_gradient_y.masked_fill_(less_location_y, 0.01)

        x_loss = torch.abs(torch.div(low_gradient_x, input_gradient_x))
        y_loss = torch.abs(torch.div(low_gradient_y, input_gradient_y))
        mut_loss = (x_loss + y_loss).mean()
        return mut_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def compute_image_gradient_o(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        grad_x = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        grad_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])

        grad_min_x = torch.min(grad_x)
        grad_max_x = torch.max(grad_x)
        grad_norm_x = torch.div((grad_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))

        grad_min_y = torch.min(grad_y)
        grad_max_y = torch.max(grad_y)
        grad_norm_y = torch.div((grad_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
        return grad_norm_x, grad_norm_y

    def compute_image_gradient(self, x):
        kernel_x = [[0, 0], [-1, 1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[0, -1], [0, 1]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        grad_x = torch.abs(F.conv2d(x, weight_x, padding=1))
        grad_y = torch.abs(F.conv2d(x, weight_y, padding=1))

        grad_min_x = torch.min(grad_x)
        grad_max_x = torch.max(grad_x)
        grad_norm_x = torch.div((grad_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))

        grad_min_y = torch.min(grad_y)
        grad_max_y = torch.max(grad_y)
        grad_norm_y = torch.div((grad_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
        return grad_norm_x, grad_norm_y

    def rgb_to_gray(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray
