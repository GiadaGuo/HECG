# Copyright (c) Facebook, Inc. and its affiliates.
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
import argparse
import os
import sys
import datetime
import time
import math
import json
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from torchvision import models as torchvision_models


import ecg_vit1 as vits
from ecg_vit1 import DINOHead

import utils
import ecg_utils
from scipy.interpolate import interp1d #用于插值

# torchvision_archs = sorted(name for name in torchvision_models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
        # choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
        #         + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

#main里用的核心函数
def train_dino(args):


    utils.init_distributed_mode(args) #初始化 PyTorch 的分布式训练环境
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    #trasform：crops列表，包括global_views和local_views
    #在transform里变成的张量
    transform = ECGDataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number
    )

    #args.data_path存放(200,12)的.npy数据
    #实例化数据增强后的数据集，可通过 DataLoader 来加载数据（有__getitem__方法）
    dataset = ecg_utils.NumpyDataset(args.data_path, transform=transform)

    #分布式训练工具：确保在多进程（多 GPU）环境中，每个进程处理的数据子集是互不重叠的
    #每个 GPU 只会看到数据的一个子集
    #shuffle=True:保证每次迭代（每个 epoch）时，数据会被洗牌(DistributedSampler非分布式好像也能用)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    #data_loader可以用enumerate输出张量
    #DataLoader 会自动把 __getitem__() 函数得到的 sample 按照batch_size组合成一个 batch！
    #每次返回的batch：每个batch包含64个样本，每个样本包含不同views，global views是(12,1,160)张量，local views是(12,1,80)张量
    data_loader = torch.utils.data.DataLoader(
        dataset, #用getitem每次会有一个[tensor,tensor,...]
        sampler=sampler, #数据的采样策略
        batch_size=args.batch_size_per_gpu, #默认值是64
        num_workers=args.num_workers, #加载数据时使用的线程数
        pin_memory=True, #数据会预先加载到锁页内存，加速后续到 GPU 的传输
        drop_last=True, #没法整除64的样本被丢弃
    )

    print(f"Data loaded: there are {len(dataset)} series.") #示例是1000//64约为15个batch

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    # 实例化student和teacher网络
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys(): #vits.__dict__.keys() 返回一个包含模块中所有属性和方法名称的列表
        student = vits.__dict__[args.arch]( #调用相应结构： vit_tiny/vit_small/vit_base
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth（在训练过程中，每个残差分支有 drop_path_rate 的概率被丢弃）
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim #根据不同VIT默认值不同，vit_small是384

    # 这个模型暂时没用上
    # if the network is a XCiT（改进的 Vision Transformer 架构）
    ## torch.hub.list：用于列出指定 GitHub 仓库中可用的模型名称
    # elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
    #     # torch.hub.load：从指定的 GitHub 仓库动态加载模型
    #     student = torch.hub.load('facebookresearch/xcit:main', args.arch,
    #                              pretrained=False, drop_path_rate=args.drop_path_rate)
    #     #加载了一个与学生模型相同架构的教师模型，但不设置随机深度（Stochastic Depth）的比率
    #     teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
    #     embed_dim = student.embed_dim

    # otherwise, we check if the architecture is in torchvision models
    # elif args.arch in torchvision_models.__dict__.keys():
    #     student = torchvision_models.__dict__[args.arch]()
    #     teacher = torchvision_models.__dict__[args.arch]()
    #     embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    # multi-crop wrapper统一处理不同分辨率的
    # 所需参数：student是backbone； DINOHead是head（head在最后统一给组合好的特征运行）
    student = ecg_utils.MultiCropWrapper(
        student, #VIT到Norm的部分
        DINOHead( #MLP和LN的部分
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        #norm_last_layer是True则不进行梯度更新（固定缩放因子）
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = ecg_utils.MultiCropWrapper(
        teacher,
        #args.out_dim默认65536；args.use_bn_in_head默认False
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # 打算在批量归一化这块用分布式训练需要做的事情
    # synchronize batch norms (if any):为了分布式？那我应该不用
    if utils.has_batchnorms(student):
        #nn.SyncBatchNorm.convert_sync_batchnorm 方法
        #将 student 和 teacher 模型中的所有普通批量归一化层（如 nn.BatchNorm2d）转换为同步批量归一化层
        #同步批量归一化（SyncBatchNorm）是一种在分布式训练中使用的批量归一化方法(允许在多个 GPU 或多个机器之间同步批量归一化的统计信息)
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        # 使用 nn.parallel.DistributedDataParallel（DDP）对 teacher 模型进行包装
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        # 原始模型对象时使用存放在teacher_without_ddp
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    # 将 student 模型的参数加载到 teacher 模型中
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    # 禁用teacher中参数的所有梯度操作
    for p in teacher.parameters():
        #teacher.parameters() ：生成器，返回模型中所有可训练的参数（权重和偏置）
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============

    #实例化对象，移到cuda
    #后续用法：loss = dino_loss(student_output, teacher_output, epoch)——参考DINOLoss类的forward
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============

    #返回一个包含两个字典（正则化参数和无正则化参数）的列表
    params_groups = utils.get_params_groups(student)
    #根据输入选择优化器
    #传入对象params_groups是参数分组（iterable of dicts defining parameter groups）
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ===========
    #初始化超参数的调度器，包括学习率（LR）、权重衰减（WD）和动量（Momentum）
    #使用余弦调度器

    # 初始化学习率调度器
    #使用余弦调度器：学习率从 scaled_lr 开始，经过预热阶段后，按照余弦曲线逐渐降低到 args.min_lr
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader), #len(data_loader)：每个 epoch 的迭代次数
        warmup_epochs=args.warmup_epochs,
    )

    #初始化权重衰减调度器
    #权重衰减系数从 args.weight_decay 开始，按照余弦曲线逐渐变化到 args.weight_decay_end
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, #初始权重衰减系数。
        args.weight_decay_end, #最终权重衰减系数。
        args.epochs, len(data_loader),
    )

    # 初始化动量调度器:动量参数从 args.momentum_teacher 开始，按照余弦曲线逐渐增加到 1
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher,
                                               1, # 最终动量值（动量参数在训练过程中逐渐增加到 1）
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    # 加载训练过程中的检查点（checkpoint），并从上次保存的状态恢复训练

    # 存储需要从检查点恢复的变量
    # 初始化 epoch 为 0，表示如果没有找到检查点文件，则从第 0 个 epoch 开始训练
    to_restore = {"epoch": 0}

    #函数定义：restart_from_checkpoint(ckp_path, run_variables=None, **kwargs)
    #效果：更新run_variables字典，添加恢复好的变量键值对
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore, #指定需要恢复的运行变量

        #这里由**kwargs传要恢复的对象？
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler, #指定混合精度训练的缩放器
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"] #获取恢复的 epoch

    start_time = time.time()
    print("Starting DINO training !")

    # 从恢复的 epoch 开始训练
    for epoch in range(start_epoch, args.epochs):

        #设置数据加载器的采样器，确保在每个 epoch 中数据的顺序是不同的
        #如果没有分布式训练，这时候 set_epoch(epoch) 仍然会影响数据的洗牌
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        # epoch在遍历上，所以可以只写1个train_one_epoch
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):


    # 定义用于记录和打印训练过程中的各种指标（如损失值、时间等）的日志工具
    metric_logger = utils.MetricLogger(delimiter="  ")


    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs) #日志输出的标题，显示当前 epoch 和总 epoch 数

    #metric_logger.log_every(self, iterable, print_freq, header=None)
    #含有yield obj代码（for obj in iterable），可以返回data_loader中对象
    #images来自data_loader出来的batch，每个batch里有batch_size个crops列表
    #(images, _)：训练时忽略target（这里要改掉，我的dataset本来就不会有target）
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration：计算全局迭代次数
        for i, param_group in enumerate(optimizer.param_groups):
            #根据全局迭代次数 it 提供对应的学习率值。
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                #根据全局迭代次数 it 提供对应的权重衰减值。
                param_group["weight_decay"] = wd_schedule[it]

        #images格式参考：images=[tensor1,tensor2,tensor3,...]=[(64,12,1,160),(64,12,1,160),(64,12,1,80),...]
        ## 按batch堆叠前每个image是[(12,1,160),(12,1,160),(12,1,80),...]——可能要注意后续解包
        # move images to gpu：
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None): #PyTorch 的自动混合精度（AMP）上下文管理器，用于在训练过程中自动切换浮点精度
            # 给teacher和student网络传数据：直接传images列表，前2个是global view后面是local
            teacher_output = teacher(images[:2])  #(128,65534);only the 2 global views pass through the teacher,images[:2]=>[(64,12,1,160),(64,12,1,160)]
            student_output = student(images) # (512,65534);images=>[(64,12,1,160),(64,12,1,160),(64,12,1,80),...]
            loss = dino_loss(student_output, teacher_output, epoch) #输出1个标量

        #检查损失值是否为有限数，如果不是，则停止训练
        ##如果损失值不是有限数，说明训练过程中可能出现了数值不稳定的情况
        if not math.isfinite(loss.item()):
            #force=True: 这个参数通常用于 print 函数，表示强制刷新输出缓冲区，确保消息立即打印出来。
            print("Loss is {}, stopping training".format(loss.item()), force=True) #loss.item()：张量标量=>正常数
            sys.exit(1) #调用 sys 模块的 exit 函数，终止程序运行，并返回退出码 1

        # student update

        optimizer.zero_grad()
        param_norms = None #用于存储梯度裁剪后的参数范数

        #.检查是否使用混合精度训练
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad: #若启用梯度裁剪
                param_norms = utils.clip_gradients(student, args.clip_grad)
            #取消最后一层的梯度更新
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()

        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad(): #禁用梯度计算的上下文
            m = momentum_schedule[it]  # momentum parameter：从设置好的调度器中取出
            #将学生网络和教师网络的参数一一对应地配对为元组
            #param_q：学生网络参数；param_k：教师网络参数
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                #param_q.detach().data: 将学生网络的参数从计算图中分离出来，避免梯度计算
                #k=m*q+(1-m)*q
                #这里momentum_schedule给的应该是文章里的lambda
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging：记录和更新日志信息，包括损失值、学习率和权重衰减参数
        torch.cuda.synchronize() #等待所有在当前 GPU 上的 CUDA 核心完成其任
        metric_logger.update(loss=loss.item())
        #optimizer.param_groups 是一个列表，其中每个元素是一个字典
        ##分组后[0]是要正则化的；[1]是不用的？
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes() #确保所有进程的统计信息一致
    print("Averaged stats:", metric_logger) # 调用 metric_logger 的 __str__ 方法，返回一个包含所有指标的字符串

    #返回一个字典，包含所有指标的全局平均值
    ##根据字典metric_logger.meters的键值对返回一个新字典
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    """
    参数输入

     dino_loss = DINOLoss(
        args.out_dim, #65534
        args.local_crops_number + 2,  # ncrops：total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    )

    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp #设定的温度系数，用于控制输出类别的区分度
        self.center_momentum = center_momentum
        self.ncrops = ncrops

        #self.center的初始化！！！
        #self.register_buffer用于注册缓冲区（buffer），通常用于注册那些不需要梯度更新的张量
        #创建缓冲区center：torch.zeros(1, out_dim) 创建一个形状为 (1, out_dim) 的零张量
        #缓冲区的名称为 "center"，可以通过 self.center 访问！
        self.register_buffer("center", torch.zeros(1, out_dim))

        # 定义了一个教师模型温度的调度策略，用于在训练过程中动态调整教师模型的温度参数（随着epoch增加先升后平）
        # 这意味着教师模型的温度系数不是不变的
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            #生成从 warmup_teacher_temp 到 teacher_temp 的warmup_teacher_temp_epochs个线性间隔值
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            #创建一个长度剩余epochs的数组，所有元素值为 teacher_temp
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    #实例化后的调用方式： loss = dino_loss(student_output, teacher_output, epoch)
    #注意loss是每个epoch计算一次的，这是特定epoch的loss
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        Input example:
            student_output:(128,65534)
            teacher_output:(512,65534)
        """

        #输入准备：
        ## student：网络输出->除以温度->打散成crops（数量=2个全局+所有局部）-> Ps(x)
        ## teacher：网络输出->中心化（减centering除以温度）-> 打散成2个crop（全局）-> Pt(x)

        #student_output train后才会得到
        student_out = student_output / self.student_temp #g(x)/T: temperature softmax输入
        #.chunk(self.ncrops)：将student_out 这个张量沿着第0个维度（默认维度）分割成 self.ncrops 个较小的张量
        # student里有所有crops，返回一个包含self.ncrops=8个张量（每个维度512/8=64）的元组
        student_out = student_out.chunk(self.ncrops) #(tensor([64,65534]),...,tensor([64,65534]))，包含8个张量

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch] #特定epoch对应的temperature

        #F代表torch.nn.functional
        #t=softmax((t-C)/tpt,dim=-1)
        #处理最后一个维度本质就是逐向量的某个维度
        #self.centre初始化为全0张量：self.register_buffer("center", torch.zeros(1, out_dim))
        #self.centre后续得到teacher_output后用update_center(self, teacher_output)更新
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # teacher中只有2个全局的crops
        teacher_out = teacher_out.detach().chunk(2) #(tensor([64,65534]),tensor([64,65534]))

        total_loss = 0
        n_loss_terms = 0
        #对每个batch里的逐样本算交叉熵，对某个样本共5+5=10对
        for iq, q in enumerate(teacher_out): #q是crops
            for v in range(len(student_out)): #student_out[v]是crops
                if v == iq: #为什么序号一样就是same view? 因为拼接顺序，student_out前面2个就是global view
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) #(64,65534):dim=-1就是逐向量每个元素（最后一维操作）
                total_loss += loss.mean() #所有 crop-pair 的交叉熵平均累加结果
                n_loss_terms += 1 #	用了多少对 crop 来计算 loss（即有多少组 teacher-student view 被对比）
        total_loss /= n_loss_terms #平均每组 teacher-student crop pair 的 loss
        self.update_center(teacher_output) #根据teacher_output来更新中心化的centre
        return total_loss #标量

    #定义更新教师模型输出中心（center）的方法
    #update_center被装饰器@torch.no_grad()装饰，将update_center函数作为参数传入torch.no_grad()
    #在update_center内部禁用梯度计算
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """

        #对 teacher_output 沿着第一个维度（dim=0）进行求和；keepdim=True保持输入输出维度一致
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) #输出(1, feature_dim)，元素是每个维度的加和
        #【分布训练操作】：将所有进程的 batch_center 进行全局归约操作，确保所有进程的 batch_center 一致
        # 确保在多 GPU 或多节点训练时，中心的更新是全局一致的。
        dist.all_reduce(batch_center)
        # 计算全局平均中心
        ## 归约后的 batch_center 除以总样本数量（当前批次的样本数量*分布式训练中的进程总数），得到全局平均中心
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        # C=m*C+(1-m)*cat([t1,t2]).mean(dim=0),m是self.center_momentum
        # batch_center=cat([t1,t2])?
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class ECGTransform:

    """

    1.随机选择窗口（global / local view）
    global窗宽设置为原始序列的80%（ws_ratio_global=0.8）=>160个点
    local窗宽设置为原始序列的40%（ws_ratio_local=0.4）=>80个点

    参考文献不是固定窗宽截的，而是随意截然后利用插值resize成想要的固定窗宽，以后要这么改吗

    2.原地增强
    1)Permutation——打乱
    global:推荐 perm_segments 值较小，例如 2~4	保留更多原始结构（减少语义破坏）
    local：推荐perm_segments 值较大，例如 6~8（甚至更多）	增强扰动强度，让模型更鲁棒
    2)window_warping——微扰动

    """

    def __init__(self,
                 scale_range=None, #如果apply_random_slice没给值scale_range也得是None
                 apply_random_slice=False,
                 ws_ratio=0.8, #默认值给的global的
                 ww_scale_range=(0.8, 1.2),
                 perm_segments=3, #global先用3，local先用7
                 target_length=160
                 ):

        self.ws_ratio = ws_ratio
        self.ww_scale_range = ww_scale_range
        self.perm_segments = perm_segments
        self.target_length=target_length
        self.apply_random_slice=apply_random_slice

        if self.apply_random_slice:
            if scale_range is None:
                raise ValueError("When apply_random_slice=True, you must provide scale_range.")
            self.scale_range = scale_range
        else:
            if scale_range is not None:
                raise ValueError("scale_range should not be provided when apply_random_slice=False.")
            self.scale_range = None


    def window_slicing(self,signal):
        """ 随机裁剪一个窗口（Window Slicing） """
        length = signal.shape[0] #200
        crop_len = int(length * self.ws_ratio)
        start = random.randint(0, length - crop_len)
        sliced = signal[start:start + crop_len]
        return sliced

    def random_slice(self,signal):
        """
        从ECG信号中按scale随机裁剪后线性插值成target长度
        signal: shape (L, C)，例如 (200, 12)
        scale_range: 裁剪比例范围
        target_length: 输出长度，与原始 global view 长度一致
        """
        L, C = signal.shape
        scale = np.random.uniform(*self.scale_range)
        crop_len = int(L * scale)

        # 起始点必须在合法范围内
        start = np.random.randint(0, L - crop_len + 1)
        cropped = signal[start:start + crop_len]

        # 插值到原始长度
        resized = np.zeros((self.target_length, C)) #用于存储重采样后的信号
        for ch in range(C):
            x_old = np.linspace(0, 1, num=crop_len) #原始信号的时间轴，用于插值的输入
            x_new = np.linspace(0, 1, num=self.target_length) #目标时间轴，用于插值的输出
            f = interp1d(x_old, cropped[:, ch], kind='cubic') #使用三次样条插值重采样
            resized[:, ch] = f(x_new)

        return resized

    def window_warping(self, signal,window_ratio=0.2):
        """
        signal: np.ndarray, shape (length, C)  e.g. (200, 12)
        超参数：window_ratio=0.2
        """
        length = signal.shape[0]
        win_len = int(length * window_ratio)
        start = random.randint(0, length - win_len)
        end = start + win_len
        warp_ratio = random.uniform(*self.ww_scale_range)

        # warp the window
        warped_len = int(win_len * warp_ratio) #变形后窗口长度：片段变形，采样点数改变
        window = signal[start:end]
        warped_window = np.interp( #对window数组每列进行插值，使其长度变为warped_len
            np.linspace(0, win_len - 1, warped_len), #插值x坐标：生成一个从0到win_len - 1的等差数列，总共有warped_len个数
            np.arange(win_len), #原窗口x坐标：生成一个从0到win_len - 1的整数数组
            window.T #需要插值的原始窗口数据
        ).T  # (warped_len, C)

        # interpolate back to original length
        warped_back = np.interp(
            np.linspace(0, warped_len - 1, win_len), np.arange(warped_len), warped_window.T
        ).T

        # replace the original window
        new_signal = np.copy(signal)
        new_signal[start:end] = warped_back
        return new_signal

    def permutation(self,signal):
        """ 将信号切成 segments 段后打乱（Permutation），保留全部数据 """
        length = signal.shape[0]
        base_len = length // self.perm_segments
        remainder = length % self.perm_segments

        # 按照“尽可能平均分”的策略生成每段的起始index:把余数的点数补给前面已整除的片段
        seg_lengths = [base_len + 1 if i < remainder else base_len for i in range(self.perm_segments)]

        #切完后存入列表大软
        segments = []
        idx = 0
        for length in seg_lengths:
            segments.append(signal[idx:idx + length])
            idx += length

        random.shuffle(segments)
        return np.concatenate(segments, axis=0)


    def __call__(self,signal):
        """
        signal: np.ndarray of shape (200, 12)
        return: torch.Tensor of shape (12,1,160)——global view;(12,1,80)——local view
        """

        view = self.window_slicing(signal) if not self.apply_random_slice else self.random_slice(signal)
        view = self.window_warping(self.permutation(view))
        view = torch.from_numpy(view).float().unsqueeze(-1).permute(1, 2, 0)

        return view

class ECGDataAugmentationDINO(object):
    def __init__(self, global_crops_scale,local_crops_scale,local_crops_number=6):
        self.global_transfo1 = ECGTransform(ws_ratio=0.8, ww_scale_range=(0.9, 1.1),perm_segments=3)
        self.global_transfo2 = ECGTransform(apply_random_slice=True,ws_ratio=0.8, ww_scale_range=(0.8, 1.2),perm_segments=4,scale_range=global_crops_scale)
        self.local_transfo = ECGTransform(apply_random_slice=True,ws_ratio=0.4,perm_segments=7,scale_range=local_crops_scale)
        self.local_crops_number = local_crops_number


    def __call__(self, signal): #当实例化后的对象被直接调用时（即用 () 语法），__call__ 会自动启用
        crops = []
        crops.append(self.global_transfo1(signal))
        crops.append(self.global_transfo2(signal))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(signal))
        return crops



if __name__ == '__main__':
    #传参用于建立路径和训练
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_dino(args)
