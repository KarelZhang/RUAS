from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'dilconv_3x3',
    'resconv_1x1',
    'resconv_3x3',
    'resdilconv_3x3',
]

IEM = Genotype(normal=[('skip_connect', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('skip_connect', 5), ('conv_3x3', 6)], normal_concat=None, reduce=None, reduce_concat=None)
NRM = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2), ('skip_connect', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('skip_connect', 6)], normal_concat=None, reduce=None, reduce_concat=None)
