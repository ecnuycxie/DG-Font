import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torch.nn.functional as F

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg

from tools.utils import *


def validateUN(data_loader, networks, epoch, args, additional=None):
    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    # switch to train mode
    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()
    # data loader
    val_dataset = data_loader['TRAINSET']
    val_loader = data_loader['VAL']

    x_each_cls = []
    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[-args.val_num:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=args.val_num, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)
    
    
    if epoch >= args.fid_start:
        # Reference guided
        with torch.no_grad():
            src = [7]
            for src_idx in src:
                for ref_idx in range(400):
                    cntt = 0
                    #依次选择src中的每张图像进行转换
                    for cnt_idx in range((int)(args.val_num/args.val_batch)):
                        x_src = x_each_cls[src_idx][args.val_batch*cnt_idx:(cnt_idx+1)*args.val_batch, :, :, :].cuda(args.gpu, non_blocking=True)
                        #随机从目标风格中选取一张图像
                        rnd_idx = torch.randperm(x_each_cls[ref_idx].size(0))[:10]
                        x_ref_rnd = x_each_cls[ref_idx][rnd_idx].cuda(args.gpu, non_blocking=True)
                        #由于val_batch我只选一张，所以这个步骤只执行一次
                        path_re = os.path.join(args.res_dir,'id_%d' %(ref_idx))
                        if not os.path.exists(path_re):
                            os.mkdir(path_re)
                        path_full = os.path.join(path_re, 'fake')
                        if not os.path.exists(path_full):
                            os.mkdir(path_full)
                        for sample_idx in range(args.val_batch):
                            #从随机排列的目标风格集选取第一张
                            x_ref_tmp = x_ref_rnd[sample_idx: sample_idx + 10].repeat((args.val_batch, 1, 1, 1))

                            c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                            s_ref = C_EMA(x_ref_tmp, sty=True)

                            s_ref = torch.mean(s_ref,dim=0)
                            s_ref = s_ref.unsqueeze(0)

                            x_res_ema_tmp,_ = G_EMA.decode(c_src, s_ref, skip1, skip2)
        
                            vutils.save_image(x_res_ema_tmp, os.path.join(path_full, '%d.png'%cntt), normalize=True,
                                        nrow=(x_res_ema_tmp.size(0) // (x_src.size(0) + 2) + 1))

                            #vutils.save_image(x_res_ema, os.path.join(args.res_dir, '{}_{}_{}{}.png'.format(args.gpu, epoch+1, src_idx, ref_idx)), normalize=True,
                            #        nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1))

                            cntt = cntt + 1

