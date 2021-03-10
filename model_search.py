import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class SearchBlock(nn.Module):

    def __init__(self, channel):
        super(SearchBlock, self).__init__()

        self.stride = 1
        self.channel = channel

        self.dc = self.distilled_channels = self.channel  # // 2
        self.rc = self.remaining_channels = self.channel
        self.c1_d = MixedOp(self.channel, self.dc)
        self.c1_r = MixedOp(self.channel, self.rc)
        self.c2_d = MixedOp(self.channel, self.dc)
        self.c2_r = MixedOp(self.channel, self.rc)
        self.c3_d = MixedOp(self.channel, self.dc)
        self.c3_r = MixedOp(self.channel, self.rc)
        self.c4 = MixedOp(self.channel, self.dc)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(self.dc * 4, self.channel, 1)

    def forward(self, input, weights):
        distilled_c1 = self.act(self.c1_d(input, weights[0]))
        r_c1 = (self.c1_r(input, weights[1]))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1, weights[2]))
        r_c2 = (self.c2_r(r_c1, weights[3]))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2, weights[4]))
        r_c3 = (self.c3_r(r_c2, weights[5]))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3, weights[6]))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)

        return out_fused



class IEM(nn.Module):
    def __init__(self, channel):
        super(IEM, self).__init__()
        self.channel = channel

        self.cell = SearchBlock(self.channel)
        self.activate = nn.Sigmoid()

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input_y, input_u, k, enhance_weights):
        if k == 0:
            t_hat = self.max_operation(input_y)
        else:
            t_hat = self.max_operation(input_u) - 0.5 * (input_u - input_y)
        t = t_hat
        t = self.cell(t, enhance_weights)
        t = self.activate(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(input_y / t, 0.0, 1.0)

        return u, t


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin
        self.channel = channel

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(self.channel))

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input, weights):
        t_list = []
        u_list = []
        u = torch.ones_like(input)
        for i in range(self.iem_nums):
            u, t = self.iems[i](input, u, i, weights[0])
            u_list.append(u)
            t_list.append(t)
        return u_list, t_list


class DenoiseNetwork(nn.Module):

    def __init__(self, layers, channel):
        super(DenoiseNetwork, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        self.stem = conv_layer(3, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel))
        self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input, weights):

        feat = self.stem(input)
        for i in range(self.nrm_nums):
            feat = self.nrms[i](feat, weights[0])
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
        self._initialize_alphas()

        self.enhance_net = EnhanceNetwork(iteratioin=self.iem_nums, channel=self.enhance_channel)
        self.denoise_net = DenoiseNetwork(layers=self.nrm_nums, channel=self.denoise_channel)

    def _initialize_alphas(self):
        k_enhance = 7  # sum(1 for i in range(self.illu_layers))
        k_denoise = 7  # sum(1 for i in range(self.alph_layers))
        num_ops = len(PRIMITIVES)
        self.alphas_enhances = []
        self.alphas_denoises = []
        for i in range(self.iem_nums):
            a = Variable(1e-3 * torch.randn(k_enhance, num_ops).cuda(), requires_grad=True)
            self.alphas_enhances.append(a)
        for i in range(self.nrm_nums):
            a = Variable(1e-3 * torch.randn(k_denoise, num_ops).cuda(), requires_grad=True)
            self.alphas_denoises.append(a)

    def new(self):
        model_new = Network().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        enhance_weights = []
        denoise_weights = []
        for i in range(self.iem_nums):
            enhance_weights.append(F.softmax(self.alphas_enhances[i], dim=-1))
        for i in range(self.nrm_nums):
            denoise_weights.append(F.softmax(self.alphas_denoises[i], dim=-1))

        u_list, t_list = self.enhance_net(input, enhance_weights)
        u_d, _ = self.denoise_net(u_list[-1], denoise_weights)
        u_list.append(u_d)

        return u_list, t_list

    def _loss(self, input, target):
        u_list, t_list = self(input)
        enhance_loss = self._criterion(input, u_list, t_list)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return enhance_loss + denoise_loss

    def _enhance_arch_loss(self, input, target):
        u_list, t_list = self(input)
        enhance_loss = self._criterion(input, u_list, t_list)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return enhance_loss + denoise_loss

    def _enhcence_loss(self, input, target):
        u_list, t_list = self(input)
        enhance_loss = self._criterion(input, u_list, t_list)
        return enhance_loss

    def _denoise_arch_loss(self, input, target):
        u_list, t_list = self(input)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return denoise_loss

    def _denoise_loss(self, input, target):
        u_list, t_list = self(input)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return denoise_loss

    def arch_parameters(self):
        return [self.alphas_enhances[0]] + [self.alphas_denoises[0]]

    def enhance_arch_parameters(self):
        return [self.alphas_enhances[0]]

    def denoise_arch_parameters(self):
        return [self.alphas_denoises[0]]

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

    def genotype(self, i, task=''):
        def _parse(weights, layers):
            gene = []
            for i in range(layers):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                gene.append((PRIMITIVES[k_best], i))
            return gene
        if task == 'enhance':
            gene = _parse(F.softmax(self.alphas_enhances[i], dim=-1).data.cpu().numpy(), 7)
        elif task == 'denoise':
            gene = _parse(F.softmax(self.alphas_denoises[i], dim=-1).data.cpu().numpy(), 7)
        genotype = Genotype(
            normal=gene, normal_concat=None,
            reduce=None, reduce_concat=None
        )
        return genotype


class DenoiseLossFunction(nn.Module):
    def __init__(self):
        super(DenoiseLossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        return 0.0000001*self.l2_loss(output, target) + self.tv_loss(output)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

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
        self.input = input  # self.rgb2yCbCr(input)
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
