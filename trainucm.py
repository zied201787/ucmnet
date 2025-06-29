import torch
import argparse
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *
import torch.optim as optim
import archs_ucm_v2
import losses
from engineucm import *
import os
import sys
from torch.optim import lr_scheduler
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils_ucm import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

ARCH_NAMES = archs_ucm_v2.__all__
LOSS_NAMES = losses.__all__


def main(config, args):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, f'checkpoints_{args.loss}')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)
    print(f"[DEBUG] args.data = {args.data}")

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    if args.data == 'ISIC2017':
        data_path = './data/ISIC2017/'
    elif args.data == 'ISIC2018':
        data_path = './data/ISIC2018/'
    elif args.data == 'PH2':
        data_path = ''
    else:
        raise Exception(f"[ERROR] Dataset '{args.data}' is not valid!")

    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data=data_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data=data_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers, drop_last=True)
    test_dataset = isic_loader(path_Data=data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=config.num_workers, drop_last=True)

    print('#----------Prepareing Models----------#')
    model = archs_ucm_v2.__dict__['UCM_NetV2'](1, 3, False)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = losses.__dict__[args.loss]().cuda()
    scaler = GradScaler()

    print('#----------Set other params----------#')
    max_miou = 0
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}

        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_key = 'module.' + k  # Ajouter 'module.' au début
            else:
                new_key = k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss = checkpoint['min_loss']
        min_epoch = checkpoint['min_epoch']
        loss = checkpoint['loss']

        logger.info(f'Resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}')
    else:
        print(f'No checkpoint found at {resume_model}, training from scratch.')

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch,
                        logger, config, scaler=scaler, epoch_num=config.epochs)

        loss, miou = val_one_epoch(val_loader, model, criterion, epoch, logger, config, epoch_num=config.epochs)

        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))

        # remove 'module.' prefix if necessary
        new_state_dict = {}
        for k, v in best_weight.items():
            if k.startswith('module.'):
                new_key = k.replace('module.', '', 1)
            else:
                new_key = k
            new_state_dict[new_key] = v

        model.module.load_state_dict(new_state_dict)

        loss = test_one_epoch(test_loader, model, criterion, logger, config, 1, 1)

        shutil.copy(os.path.join(checkpoint_dir, 'best.pth'),
                    os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='GT_BceDiceLoss_new2', choices=LOSS_NAMES, help='Loss function')
    parser.add_argument('--data', type=str, default='ISIC2017', help='datasets')
    args = parser.parse_args()

    config = setting_config
    main(config, args)
