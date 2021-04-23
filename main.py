import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet
from models.inception import InceptionV3

from train.train import trainGAN

from validation.validation import validateUN

from tools.utils import *
from datasets.datasetgetter import get_dataset
from tools.ops import initialize_queue

from tensorboardX import SummaryWriter

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--data_path', type=str, default='../data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--epochs', default=250, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--val_num', default=190, type=int,help='Number of test images for each style')
parser.add_argument('--val_batch', default=10, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=80, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='Call for valiation only mode')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8993', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')
parser.add_argument('--w_off', default=0.5, type=float, help='Coefficient of offset normalization. loss of G')


def main():
    ####################
    # Default settings #
    ####################
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0


    args.train_mode = 'GAN'


    den = args.iters//args.iters

    # unsup_start : train networks with supervised data only before unsup_start
    # separated : train IIC only until epoch = args.separated
    # ema_start : Apply EMA to Generator after args.ema_start


    args.unsup_start = 0
    args.separated = 0
    args.ema_start = 1
    args.fid_start = 1

    args.unsup_start = args.unsup_start // den
    args.separated = args.separated // den
    args.ema_start = args.ema_start // den
    args.fid_start = args.fid_start // den

    # Cuda Set-up
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.multiprocessing_distributed = False

    if len(args.gpu) > 1:
        args.multiprocessing_distributed = True
    print(args.multiprocessing_distributed)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args.distributed)

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    print("MULTIPROCESSING DISTRIBUTED : ", args.multiprocessing_distributed)

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.idea', '.git', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if len(args.gpu) == 1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)

    # # of GT-classes
    args.num_cls = args.output_k

    # Classes to use
    args.att_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399]

    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # Logging
    logger = SummaryWriter(args.event_dir)

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)
    cudnn.benchmark = True

    # get dataset and data loader
    train_dataset, val_dataset = get_dataset(args)
    train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})

    # map the functions to execute - un / sup / semi-
    trainFunc, validationFunc = map_exec_func(args)

    # print all the argument
    print_args(args)

    # All the test is done in the training - do not need to call
    if args.validation:
        validationFunc(val_loader, networks, 999, args, {'logger': logger})
        return

    # For saving the model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        for arg in vars(args):
            record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
        record_txt.close()

    # Run
    #validationFunc(val_loader, networks, 0, args, {'logger': logger, 'queue': queue})

    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch+1))
        if (epoch + 1) % (args.epochs // 25) == 0:
            save_model(args, epoch, networks, opts)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch == args.ema_start and 'GAN' in args.train_mode:
            if args.distributed:
                networks['G_EMA'].module.load_state_dict(networks['G'].module.state_dict())
            else:
                networks['G_EMA'].load_state_dict(networks['G'].state_dict())

        trainFunc(train_loader, networks, opts, epoch, args, {'logger': logger})

        validationFunc(val_loader, networks, epoch, args, {'logger': logger})

#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDG'

    networks = {}
    opts = {}
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'D' in args.to_train:
        networks['D'] = Discriminator(args.img_size, num_domains=args.output_k)
    if 'G' in args.to_train:
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)

    if args.distributed:
        if args.gpu is not None:
            print('Distributed to', args.gpu)
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            args.workers = int(args.workers / args.ngpus_per_node)
            for name, net in networks.items():
                if name in ['inceptionNet']:
                    continue
                net_tmp = net.cuda(args.gpu)
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[args.gpu], output_device=args.gpu)
        else:
            for name, net in networks.items():
                net_tmp = net.cuda()
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for name, net in networks.items():
            networks[name] = net.cuda(args.gpu)
    else:
        for name, net in networks.items():
            networks[name] = torch.nn.DataParallel(net).cuda()

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C'].module.parameters() if args.distributed else networks['C'].parameters(),
            1e-4, weight_decay=0.001)
        if args.distributed:
            networks['C_EMA'].module.load_state_dict(networks['C'].module.state_dict())
        else:
            networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(
            networks['D'].module.parameters() if args.distributed else networks['D'].parameters(),
            1e-4, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].module.parameters() if args.distributed else networks['G'].parameters(),
            1e-4, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            if not args.multiprocessing_distributed:
                for name, net in networks.items():
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                    if 'module' in tmp_keys:
                        tmp_new_dict = OrderedDict()
                        for key, val in checkpoint[name + '_state_dict'].items():
                            tmp_new_dict[key[7:]] = val
                        net.load_state_dict(tmp_new_dict)
                        networks[name] = net
                    else:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                        networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))


def get_loader(args, dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    print(len(val_dataset))

    train_dataset_ = train_dataset['TRAIN']
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size,
                                                shuffle=(train_sampler is None), num_workers=args.workers,
                                                pin_memory=True, sampler=train_sampler, drop_last=False)


    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)

    val_loader = {'VAL': val_loader, 'VALSET': val_dataset, 'TRAINSET': train_dataset['FULL']}

    return train_loader, val_loader, train_sampler


def map_exec_func(args):
    if args.train_mode == 'GAN':
        trainFunc = trainGAN
        validationFunc = validateUN


    return trainFunc, validationFunc


def save_model(args, epoch, networks, opts):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        # if (epoch + 1) % (args.epochs//10) == 0:
        with torch.no_grad():
            save_dict = {}
            save_dict['epoch'] = epoch + 1
            for name, net in networks.items():
                save_dict[name+'_state_dict'] = net.state_dict()
                if name in ['G_EMA', 'C_EMA']:
                    continue
                save_dict[name.lower()+'_optimizer'] = opts[name].state_dict()
            print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
            save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
        check_list.close()


if __name__ == '__main__':
    main()
