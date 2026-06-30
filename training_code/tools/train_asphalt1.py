"""TRAINING CODE FOR ASPHALT CRACK SEGMENTATION
This script trains a segmentation model (e.g., UNetPP) on the asphalt crack dataset.
It includes data loading, model creation, training loop, evaluation, and checkpointing.
"""
import os
import argparse
import time
from pathlib import Path
import datetime
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
from train_utils.train_and_eval_asphalt import evaluate, create_lr_scheduler, train_one_epoch_loss
from train_utils.my_dataset import CrackDataset, SegmentationPresetTrain, SegmentationPresetEval
from train_utils.utils import plot, show_config

# from models.segformer.segformer import SegFormer
# from models.unet.unet import UNet
# from models.unet.mobilenet_unet import MobileV3Unet
# from models.unet.vgg_unet import VGG16UNet
# from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
# from models.fcn.fcn import fcn_resnet50
# from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.unet.UnetPP import UNetPP
from models.unet.UnetPP_backbone import build_unetpp_model

# from models.dinov3.dinov3 import DINODeepLab


project_root_ = Path(__file__).resolve().parent.parent.parent
OUTPUT_SAVE_PATH = project_root_ / 'weights' / 'Unet30June'  # Change this to your desired output path
model_name = "30junetpp"  # Change this to your desired model name
os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_SAVE_PATH / "latest_checkpoint.pth"

counts_file = project_root_ / "weights" / "class_counts.pt"


def get_transform(train, mean=(0.487, 0.487, 0.487), std=(0.145, 0.145, 0.145)):
    img_size = 512
    if train:
        return SegmentationPresetTrain(img_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(img_size, mean=mean, std=std)


def create_model(aux, num_classes, pretrained=True):
    # model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    # model = fcn_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = deeplabv3_resnet101(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = deeplabv3_mobilenetv3_large(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = SegFormer(num_classes=num_classes, phi=args.phi, pretrained=args.pretrained)
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = MobileV3Unet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    # model = VGG16UNet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    # model = DINODeepLab(num_classes=num_classes, backbone_name="dinov2_vitl14")
    model = UNetPP(in_channels=3, num_classes=num_classes, deep_supervision=True, base_channels=64)
    # model = build_unetpp_model(
    #     encoder="resnet50",   # or efficientnet_b3
    #     pretrained=pretrained,
    #     in_channels=3,
    #     num_classes=num_classes,
    #     dec_ch=320,
    #     use_se=True,
    #     use_attn_gates=True,
    #     deep_supervision=True
    # )
    return model


def save_checkpoint(save_path, epoch, model, optimizer, lr_scheduler, scaler, best_dice, train_loss, dice_coefficient):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_dice": best_dice,
        "train_loss": train_loss,
        "dice_coefficient": dice_coefficient
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, save_path)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    mean = (0.4766, 0.4766, 0.4766)  # 478
    std = (0.1454, 0.1454, 0.1454)  # 145

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_dataset = CrackDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = CrackDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)

    model = create_model(aux=args.aux, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    if counts_file.exists():
        class_counts = torch.load(counts_file)
        print("\nLoaded class counts:")
        print(class_counts)
    else:
        print("\nCalculating class distribution...")

        count_loader = DataLoader(train_dataset, batch_size=args.batch_size , shuffle=False,
                                  num_workers=num_workers, pin_memory=True, collate_fn=train_dataset.collate_fn)

        class_counts = torch.zeros(num_classes, dtype=torch.long)

        for _, masks in tqdm(count_loader, desc="Computing class counts"):
            masks = masks.view(-1)
            valid = masks != 255
            hist = torch.bincount( masks[valid], minlength=num_classes)
            class_counts += hist.cpu()
        # print("\nClass Counts:")
        # print(class_counts)
        torch.save(class_counts, counts_file)
        print(f"\nSaved class counts to "f"{counts_file}")

    if args.pretrained_weights != "":
        assert os.path.exists(args.pretrained_weights), ("weights file: '{}' not exist."
                                                         .format(args.pretrained_weights))
        model_dict = model.state_dict()
        checkpoint = torch.load(args.pretrained_weights, map_location=device)        
        # Handle both raw state_dict and dict with "state_dict"
        if "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        print("load_key: ", load_key)
        print("no_load_key: ", no_load_key)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = {
        'adam': torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                 weight_decay=args.weight_decay),
        'adamw': torch.optim.AdamW(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                   weight_decay=args.weight_decay),
        'sgd': torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    }[args.optimizer_type]
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True,
                                       warmup_epochs=args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    best_dice = 0.0
    train_loss = []
    dice_coefficient = []
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        train_loss = checkpoint.get("train_loss", [])
        dice_coefficient = checkpoint.get("dice_coefficient", [])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resuming from epoch {args.start_epoch}, "f"best dice={best_dice:.4f}")

    results_file = OUTPUT_SAVE_PATH / "{}-results.txt".format(model_name)
    config_info = {
        'device': args.device,
        'data_path': args.data_path,
        'num_classes': num_classes,
        'model': model.__class__.__name__,
        'backbone_pretrained': args.pretrained,
        'pretrained_weights': args.pretrained_weights,
        "loss": "cross_entropy(weight=[1.0,2.0])+dice_loss",
        'optimizer_type': args.optimizer_type,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'img_size': '1024 * 419',
        'start_epoch': args.start_epoch,
        'epochs': args.epochs,
        "warmup_epochs": args.warmup_epochs,
        'weights_save_best': args.save_best,
        'amp': args.amp,
        'num_workers': num_workers
    }

    show_config(config_info)

    with open(results_file, "a") as f:
        f.write("Configurations:\n")
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n\n")

    img_save_path = OUTPUT_SAVE_PATH / f"{model_name}_training_curve.png"

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        mean_loss, lr = train_one_epoch_loss(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            num_classes,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler,
            grad_clip_norm=1.0
        )
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)

        train_loss.append(mean_loss)
        dice_coefficient.append(dice)
        plot(
            train_loss,
            dice_coefficient,
            img_save_path
        )
        print(f"MEAN LOSS: {mean_loss:.3f}")
        print("VALINFO", val_info)
        print(f"dice coefficient: {dice:.3f}")

        epoch_end_time = time.time()
        one_epoch_time = epoch_end_time - epoch_start_time
        one_epoch_time = str(datetime.timedelta(seconds=int(one_epoch_time)))
        print(f"training epoch {epoch} time {one_epoch_time}")
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.8f}\n" \
                         f"dice coefficient: {dice:.3f}\n" \
                         f"epoch time: {one_epoch_time}\n"

            f.write(train_info + val_info + "\n\n")

            torch.save(model.state_dict(), OUTPUT_SAVE_PATH / f"{model_name}_best_epoch{epoch}_dice{dice:.3f}.pth")
            best_model_info = OUTPUT_SAVE_PATH / f"{model_name}_best_epoch{epoch}_dice{dice:.3f}.txt"
            with open(best_model_info, "w") as f:
                f.write(train_info + val_info)

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
                torch.save(model.state_dict(), OUTPUT_SAVE_PATH / f"{model_name}_best_epoch{epoch}_dice{dice:.3f}.pth")
                best_model_info = OUTPUT_SAVE_PATH / f"{model_name}_best_epoch{epoch}_dice{dice:.3f}.txt"
                with open(best_model_info, "w") as f:
                    f.write(train_info + val_info)
            else:
                continue
        save_checkpoint(
            save_path=CHECKPOINT_FILE, epoch=epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
            scaler=scaler, best_dice=best_dice, train_loss=train_loss, dice_coefficient=dice_coefficient)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    """
Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("--data-path",
                        default=r"G:\Devendra\ASPHALT\NEWTRAINING\SPLIT",
                        help="root")
    parser.add_argument("--num-classes", default=5, type=int)  # exclude background
    parser.add_argument("--aux", default=True, type=bool, help="deeplabv3 auxilier loss")
    parser.add_argument("--phi", default="b0", help="Use backbone")
    parser.add_argument('--pretrained', default=True, type=bool, help='backbone')
    parser.add_argument('--pretrained-weights', type=str,
                        default=r"Y:\Devendra_Files\segmentation_training\weights\Unet29June\29junetpp_best_epoch33_dice0.675.pth",
                        help='pretrained weights path')
    parser.add_argument('--optimizer-type', default="adamw")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')  # 0.00006
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')

    parser.add_argument('--save-best', default=False, type=bool, help='only save best dice weights')
    parser.add_argument('--resume', default=str(CHECKPOINT_FILE), help='resume from checkpoint')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for automatic mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
