import torch
import torch.nn as nn

OPS = {
    'avg_pool_3x3': lambda C_in, C_out: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C_in, C_out: nn.MaxPool2d(3, stride=1, padding=1),
    'skip_connect': lambda C_in, C_out: Identity(),
    'conv_1x1': lambda C_in, C_out: ConvBlock(C_in, C_out, 1),
    'conv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3),
    'conv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5),
    'conv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7),
    'dilconv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3, dilation=2),
    'dilconv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5, dilation=2),
    'dilconv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7, dilation=2),
    'resconv_1x1': lambda C_in, C_out: ResBlock(C_in, C_out, 1),
    'resconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3),
    'resconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5),
    'resconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7),
    'resdilconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3, dilation=2),
    'resdilconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5, dilation=2),
    'resdilconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7, dilation=2),
}

class ConvBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.op(x)

class ResBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                            groups=groups)

    def forward(self, x):
        return self.op(x) + x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


