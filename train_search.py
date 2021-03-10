import os
import sys
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.autograd import Variable
from model_search import Network
from architect_enhance import Architect as Enhence_Architect
from architect_denoise import Architect as Denoise_Architect
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

EXP_path = r'./EXP/Cooperative-Search/'
if not os.path.isdir(EXP_path):
    os.mkdir(EXP_path)
inference_dir = EXP_path + '/inference/'
if not os.path.isdir(inference_dir):
    os.mkdir(inference_dir)
model_path = EXP_path + '\model/'
if not os.path.isdir(model_path):
    os.mkdir(model_path)
arch_path = EXP_path + '/arch/'
if not os.path.isdir(arch_path):
    os.mkdir(arch_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(EXP_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model = Network()
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum)

    optimizer_enhance = torch.optim.SGD(
        model.enhance_net_parameters(),
        args.learning_rate,
        momentum=args.momentum)

    optimizer_denoise = torch.optim.SGD(
        model.denoise_net_parameters(),
        args.learning_rate,
        momentum=args.momentum)

    # prepare DataLoader
    train_low_data_names = r'D:\ZJA\data\LOL\trainA/*.png'
    # train_low_data_names = r'H:\image-enhance\UPE500\trainA/*.png'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    valid_low_data_names = r'D:\ZJA\data\LOL\validA/*.png'
    # valid_low_data_names = r'H:\image-enhance\UPE500\validA/*.png'
    ValidDataset = MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        ValidDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    enhance_architect = Enhence_Architect(model, args)
    denoise_architect = Denoise_Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        logging.info('Architect of IEM:')

        logging.info('iem = %s', str(0))
        genotype = model.genotype(0, task='enhance')
        logging.info('genotype = %s', genotype)
        logging.info('iem %s', str(0))
        logging.info('%s', F.softmax(model.alphas_enhances[0], dim=-1))

        logging.info('Architect of NRM:')
        logging.info('nrm = %s', str(0))
        genotype = model.genotype(0, task='denoise')
        logging.info('genotype = %s', genotype)
        logging.info('nrm %s', str(0))
        logging.info('%s', F.softmax(model.alphas_denoises[0], dim=-1))

        # training
        train(train_queue, valid_queue, model, enhance_architect, denoise_architect, optimizer_enhance,
              optimizer_denoise, lr, epoch)


def train(train_queue, valid_queue, model, enhance_architect, denoise_architect, optimizer_enhance, optimizer_denoise,
          lr, epoch):
    for step, (input) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        input_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()

        enhcence_loss = 0
        enhance_architect.step(input, input, input_search, input_search, lr, optimizer_enhance, unrolled=True)
        optimizer_enhance.zero_grad()
        enhcence_loss = model._enhcence_loss(input, input)
        enhcence_loss.backward()
        nn.utils.clip_grad_norm(model.enhance_net_parameters(), args.grad_clip)
        optimizer_enhance.step()

        denoise_loss = 0
        if step % 10 == 0:
            denoise_architect.step(input, input, input_search, input_search, lr, optimizer_denoise, unrolled=True)
            optimizer_denoise.zero_grad()
            denoise_loss = model._denoise_loss(input, input)
            denoise_loss.backward()
            nn.utils.clip_grad_norm(model.denoise_net_parameters(), args.grad_clip)
            optimizer_denoise.step()

        if step % args.report_freq == 0:
            logging.info('train %03d %f', step, enhcence_loss + denoise_loss)


if __name__ == '__main__':
    main()
