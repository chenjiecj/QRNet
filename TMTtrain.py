
import datetime
import os
from functools import partial
from nets.detr_training import ModelEMA
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from nets.QRNet import QRNet
from nets.detr_training import (build_loss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DetrDataset, detr_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


from nets.yolo import YoloBody
from nets.yolo_training import (LossYOLO, ModelEMAYOLO, get_lr_schedulerYOLO,
                                set_optimizer_lrYOLO, weights_init)
from utilsYOLO.callbacks import EvalCallbackYOLO, LossHistoryYOLO
from utilsYOLO.dataloader import YoloDataset, yolo_dataset_collate
from utilsYOLO.utils import (download_weights, get_classes, seed_everything,
                             show_config, worker_init_fn)
from utilsYOLO.utils_fit import fit_one_epochYOLO

if __name__ == "__main__":
    Cuda = True
    seed = 3407
    distributed = False
    fp16 = True
    classes_path = 'model_data/voc_classes.txt'
    model_path = ''
    input_shape = [640, 640]
    # ---------------------------------------------#
    #   resnet50
    #   qrnet
    #   efficientnet
    #   mobilenet
    # ---------------------------------------------#
    backbone = "qrnet"
    pretrained = False

    Init_Epoch = 0
    Freeze_Epoch = 0
    UnFreeze_Epoch = 520
    ratio             = 0.38461538
    Unfreeze_batch_sizeYOLO = 32
    Freeze_batch_sizeYOLO = 32
    Unfreeze_batch_size = 32
    Freeze_batch_size = 32
    Freeze_Train = False
    Init_lrYOLO = 1e-2
    Min_lrYOLO = Init_lrYOLO * 0.01
    Init_lr = 16e-4
    optimizer_typeYOLO = "sgd"
    momentumYOLO = 0.937
    optimizer_type = "adamw"
    momentum = 0.90
    weight_decayYOLO = 5e-4
    weight_decay = 1e-4
    lr_decay_type = "cos"
    save_period = 10
    save_dir = 'logs'

    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.27

    eval_flag = True
    eval_period = 10
    aux_loss = True
    num_workers = 4
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    seed_everything(seed)


    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    class_names, num_classes = get_classes(classes_path)

    modelYOLO = YoloBody(backbone, num_classes,  pretrained=pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = modelYOLO.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        modelYOLO.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mPlease ignore the UserWarning: Detected call of lr_scheduler.step() before optimizer.step() The MultiStep learning rate scheduler milestone is 1000 and the learning rate does not decay\033[0m")

    yolo_loss = LossYOLO(modelYOLO)
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_diryolo = os.path.join(save_dir, "loss_" + str(time_str))
        loss_historyYOLO = LossHistoryYOLO(log_diryolo, modelYOLO, input_shape=input_shape)
    else:
        loss_historyYOLO = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_trainYOLO = modelYOLO.train()

    if Cuda:
        if distributed:
            model_train = model_trainYOLO.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_trainYOLO = torch.nn.DataParallel(model_trainYOLO)
            cudnn.benchmark = True
            model_trainYOLO = model_trainYOLO.cuda()

    emaYOLO = ModelEMAYOLO(model_trainYOLO)
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_sizeYOLO=Freeze_batch_sizeYOLO, Unfreeze_batch_sizeYOLO=Unfreeze_batch_sizeYOLO, Freeze_Train=Freeze_Train, \
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,\
            Init_lr=Init_lr, Init_lrYOLO=Init_lrYOLO,optimizer_type=optimizer_type, optimizer_typeYOLO=optimizer_typeYOLO,momentum=momentum,
            momentumYOLO=momentumYOLO,
            lr_decay_typeYOLO=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train,
            num_val=num_val
        )
        print(
            "\n\033[1;33;44mPlease ignore the UserWarning: Detected call of lr_scheduler.step() before optimizer.step() The MultiStep learning rate scheduler milestone is 1000 and the learning rate does not decay\033[0m")
        wanted_step = 5e4 if optimizer_typeYOLO == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_sizeYOLO * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_sizeYOLO == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')


    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in modelYOLO.backbone.parameters():
                param.requires_grad = False

        batch_sizeYOLO = Freeze_batch_sizeYOLO if Freeze_Train else Unfreeze_batch_sizeYOLO

        nbs = 64
        lr_limit_maxYOLO = 1e-3 if optimizer_typeYOLO == 'adam' else 5e-2
        lr_limit_minYOLO = 3e-4 if optimizer_typeYOLO == 'adam' else 5e-4
        Init_lr_fitYOLO = min(max(batch_sizeYOLO / nbs * Init_lrYOLO, lr_limit_minYOLO), lr_limit_maxYOLO)
        Min_lr_fitYOLO = min(max(batch_sizeYOLO / nbs * Min_lrYOLO, lr_limit_minYOLO * 1e-2), lr_limit_maxYOLO * 1e-2)

        pg0, pg1, pg2 = [], [], []
        for k, v in modelYOLO.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizerYOLO = {
            'adam': optim.Adam(pg0, Init_lr_fitYOLO, betas=(momentumYOLO, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fitYOLO, momentum=momentumYOLO, nesterov=True)
        }[optimizer_typeYOLO]
        optimizerYOLO.add_param_group({"params": pg1, "weight_decay": weight_decayYOLO})
        optimizerYOLO.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_schedulerYOLO(lr_decay_type, Init_lr_fitYOLO, Min_lr_fitYOLO, UnFreeze_Epoch)

        epoch_step = num_train // batch_sizeYOLO
        epoch_step_val = num_val // batch_sizeYOLO

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if emaYOLO:
            emaYOLO.updates = epoch_step * Init_Epoch

        train_datasetYOLO = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                    mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                    train=True, special_aug_ratio=special_aug_ratio)
        val_datasetYOLO = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                  special_aug_ratio=0)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasetYOLO, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_datasetYOLO, shuffle=False, )
            batch_sizeYOLO = batch_sizeYOLO // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        genYOLO = DataLoader(train_datasetYOLO, shuffle=shuffle, batch_size=batch_sizeYOLO, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_valYOLO = DataLoader(val_datasetYOLO, shuffle=shuffle, batch_size=batch_sizeYOLO, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callbackYOLO = EvalCallbackYOLO(modelYOLO, input_shape, class_names, num_classes, val_lines, log_diryolo, Cuda, \
                                             eval_flag=eval_flag, period=eval_period)
        else:
            eval_callbackYOLO = None




    model = QRNet(backbone, num_classes, aux_loss=aux_loss)

    detr_loss = build_loss(num_classes)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            detr_loss = detr_loss.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            detr_loss = detr_loss.cuda()

    ema = ModelEMA(model_train)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        if optimizer_type in ['adam', 'adamw']:
            Init_lr_fit = Init_lr
        else:
            nbs = 64
            lr_limit_max = 5e-2
            lr_limit_min = 5e-4
            Init_lr_fit = min(max(Unfreeze_batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)


        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n],
                "lr": Init_lr_fit / 10,
            },
        ]

        optimizer = {
            'adam': optim.Adam(param_dicts, Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adamw': optim.AdamW(param_dicts, Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(param_dicts, Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay),
        }[optimizer_type]
        lr_scale_ratio = [1, 0.1]

        epoch_step = num_train // Unfreeze_batch_size
        epoch_step_val = num_val // Unfreeze_batch_size

        train_dataset = DetrDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = DetrDataset(val_lines, input_shape, num_classes, train=False)


        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = Unfreeze_batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=Unfreeze_batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=detr_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Unfreeze_batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=detr_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))


        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None



        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch <= int(UnFreeze_Epoch*ratio-1):
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    nbs = 64
                    lr_limit_maxYOLO = 1e-3 if optimizer_typeYOLO == 'adam' else 5e-2
                    lr_limit_minYOLO = 3e-4 if optimizer_typeYOLO == 'adam' else 5e-4
                    Init_lr_fitYOLO = min(max(batch_sizeYOLO / nbs * Init_lrYOLO, lr_limit_minYOLO), lr_limit_maxYOLO)
                    Min_lr_fitYOLO = min(max(batch_sizeYOLO / nbs * Min_lrYOLO, lr_limit_minYOLO * 1e-2), lr_limit_maxYOLO * 1e-2)

                    lr_scheduler_func = get_lr_schedulerYOLO(lr_decay_type, Init_lr_fitYOLO, Min_lr_fitYOLO, UnFreeze_Epoch)

                    for param in model.backbone.parameters():
                        param.requires_grad = True

                    epoch_step = num_train // batch_sizeYOLO
                    epoch_step_val = num_val // batch_sizeYOLO

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    if emaYOLO:
                        emaYOLO.updates = epoch_step * epoch

                    if distributed:
                        batch_size = batch_sizeYOLO // ngpus_per_node

                    genYOLO = DataLoader(train_datasetYOLO, shuffle=shuffle, batch_size=Unfreeze_batch_sizeYOLO, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                    gen_valYOLO = DataLoader(val_datasetYOLO, shuffle=shuffle, batch_size=Unfreeze_batch_sizeYOLO, num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                    UnFreeze_flag = True

                genYOLO.dataset.epoch_now = epoch
                gen_valYOLO.dataset.epoch_now = epoch

                if distributed:
                    train_sampler.set_epoch(epoch)

                set_optimizer_lrYOLO(optimizerYOLO, lr_scheduler_func, epoch)

                fit_one_epochYOLO(model_trainYOLO, modelYOLO, emaYOLO, yolo_loss, loss_historyYOLO, eval_callbackYOLO, optimizerYOLO, epoch, epoch_step,
                              epoch_step_val, genYOLO, gen_valYOLO, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                              local_rank)

                if distributed:
                    dist.barrier()


            else:
                if epoch == int(UnFreeze_Epoch*ratio):
                    model_dict = model.state_dict()
                    pretrained_dict = torch.load('logs/last_epoch_weights.pth', map_location=device)
                    load_key, no_load_key, temp_dict = [], [], {}
                    new_dict = []
                    new_pretrained_dict = {}
                    # new_pretrained_dict2 = {}
                    for k, v in pretrained_dict.items():
                        if "backbone" in k:
                            new_dict.append(k)
                    for k, v in pretrained_dict.items():
                        if k in new_dict:
                            new_pretrained_dict[k] = v
                    for k, v in new_pretrained_dict.items():
                        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                            temp_dict[k] = v
                            load_key.append(k)
                        else:
                            no_load_key.append(k)
                    model_dict.update(temp_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:

                    for param in model.backbone.parameters():
                        param.requires_grad = True

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    if distributed:
                        batch_size = batch_size // ngpus_per_node

                    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=detr_dataset_collate, sampler=train_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=True, collate_fn=detr_dataset_collate, sampler=val_sampler,
                                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                    UnFreeze_flag = True

                if distributed:
                    train_sampler.set_epoch(epoch)
                torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1).step()
                fit_one_epoch(model_train, model, ema, detr_loss, loss_history, eval_callback, optimizer, epoch,
                              epoch_step,
                              epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                              local_rank)

                if distributed:
                    dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
