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
from train_utils.train_and_eval_asphalt import evaluate, create_lr_scheduler, train_one_epoch_loss, criterion
from train_utils.my_dataset import CrackDataset, SegmentationPresetTrain, SegmentationPresetEval, COLOR_MAP
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
from clearml import Dataset
from clearml import OutputModel
from clearml import Task

# from models.dinov3.dinov3 import DINODeepLab


project_root_ = Path(__file__).resolve().parent.parent.parent
OUTPUT_SAVE_PATH = project_root_ / 'weights' / '19sept'  # Change this to your desired output path
model_name = "5oct"  # Change this to your desired model name
os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_SAVE_PATH / "latest_checkpoint.pth"

counts_file = project_root_ / "weights" / "class_counts_asphalt.pt"


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


def save_checkpoint(save_path, epoch, model, optimizer, lr_scheduler, scaler, best_dice, train_loss,
                    dice_coefficient, best_epoch=None, best_model_path=None):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_dice": best_dice,
        "train_loss": train_loss,
        "dice_coefficient": dice_coefficient,
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path.resolve()) if best_model_path is not None else None
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, save_path)


def log_segmentation_metrics(logger, split, confmat, dice, epoch, class_names):
    accuracy, precision, recall, iou, f1 = confmat.compute()
    logger.report_scalar("accuracy", split, value=accuracy.item(), iteration=epoch)
    logger.report_scalar("dice", split, value=dice, iteration=epoch)
    for name, values in (("precision", precision), ("recall", recall), ("IoU", iou), ("F1", f1)):
        for class_name, value in zip(class_names, values.tolist()):
            logger.report_scalar(f"{split}/{name}", class_name, value=value, iteration=epoch)


def pull_clearml_dataset(args, task=None):
    dataset_project = args.clearml_dataset_project or args.clearml_project
    dataset_kwargs = {}
    if args.clearml_dataset_id:
        dataset_kwargs["dataset_id"] = args.clearml_dataset_id
    else:
        if not args.clearml_dataset_name:
            raise ValueError(
                "ClearML dataset is required. Pass --clearml-dataset-id or "
                "--clearml-dataset-name."
            )
        dataset_kwargs["dataset_project"] = dataset_project
        dataset_kwargs["dataset_name"] = args.clearml_dataset_name
        if args.clearml_dataset_version:
            dataset_kwargs["dataset_version"] = args.clearml_dataset_version

    dataset = Dataset.get(**dataset_kwargs)
    data_path = dataset.get_local_copy()

    dataset_info = {
        "dataset_project": dataset_project,
        "dataset_name": args.clearml_dataset_name or "",
        "dataset_version": args.clearml_dataset_version or "",
        "dataset_id": dataset.id,
        "dataset_local_path": data_path,
    }
    if task is not None:
        task.connect(dataset_info, name="Dataset", ignore_remote_overrides=True)
        task.get_logger().report_text(
            "Using ClearML dataset "
            f"id={dataset.id}, project={dataset_info['dataset_project']}, "
            f"name={dataset_info['dataset_name']}, local_path={data_path}"
        )

    print("\nUsing ClearML dataset:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    args.data_path = data_path
    return dataset_info


def main(args, task=None):
    logger = task.get_logger() if task is not None else None
    dataset_info = pull_clearml_dataset(args, task=task)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    mean = (0.4787, 0.4787, 0.4787)  # 478
    std = (0.1472, 0.1472, 0.1472)  # 145

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
    best_dice = -1.0
    best_epoch = None
    best_model_path = None
    train_loss = []
    dice_coefficient = []
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        best_epoch = checkpoint.get("best_epoch")
        if checkpoint.get("best_model_path"):
            best_model_path = Path(checkpoint["best_model_path"])
        train_loss = checkpoint.get("train_loss", [])
        dice_coefficient = checkpoint.get("dice_coefficient", [])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resuming from epoch {args.start_epoch}, "f"best dice={best_dice:.4f}")

    results_file = OUTPUT_SAVE_PATH / "{}-results.txt".format(model_name)
    config_info = {
        'device': str(device),
        'data_path': args.data_path,
        'clearml_dataset_project': dataset_info["dataset_project"],
        'clearml_dataset_name': dataset_info["dataset_name"],
        'clearml_dataset_version': dataset_info["dataset_version"],
        'clearml_dataset_id': dataset_info["dataset_id"],
        'clearml_dataset_local_path': dataset_info["dataset_local_path"],
        'num_classes': num_classes,
        'model': model.__class__.__name__,
        'backbone_pretrained': args.pretrained,
        'pretrained_weights': args.pretrained_weights,
        "loss": repr(criterion),
        'optimizer_type': args.optimizer_type,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'img_size': '256 * 104',
        'start_epoch': args.start_epoch,
        'epochs': args.epochs,
        "warmup_epochs": args.warmup_epochs,
        'weights_save_best': args.save_best,
        'amp': args.amp,
        'num_workers': num_workers,
        'normalization_mean': list(mean),
        'normalization_std': list(std),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'ignore_index': 255,
        'grad_clip_norm': 1.0,
        'deep_supervision': True,
        'base_channels': 64,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_metrics': 'mean loss only',
        'metric_scale': '0 to 1; existing confusion matrix epsilon conventions',
        'best_model_metric': 'validation Dice (foreground, batch mean)'
    }

    class_labels = {index: name for index, name in COLOR_MAP.values()}
    class_names = [class_labels.get(index, f"class_{index}") for index in range(num_classes)]
    best_output_model = None
    if task is not None:
        

        task.connect(config_info, name="Runtime", ignore_remote_overrides=True)
        best_output_model = OutputModel(
            task=task, name=f"{model_name}_best_val_dice", framework="PyTorch",
            config_dict=config_info,
            label_enumeration={name: index for index, name in enumerate(class_names)})
        if best_model_path is not None and best_model_path.is_file():
            best_output_model.update_weights(
                weights_filename=str(best_model_path), auto_delete_file=False,
                iteration=best_epoch, async_enable=False)
            logger.report_single_value("best_val_dice", best_dice)
            if best_epoch is not None:
                logger.report_single_value("best_val_epoch", best_epoch)
        elif args.start_epoch > 0:
            print("Previous best weights are unavailable; ClearML will upload the next improved validation model.")

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
        print(f"val dice coefficient: {dice:.3f}")

        epoch_end_time = time.time()
        one_epoch_time = epoch_end_time - epoch_start_time
        if logger is not None:
            logger.report_scalar("loss", "train", value=mean_loss, iteration=epoch)
            logger.report_scalar("learning rate", "train", value=lr, iteration=epoch)
            logger.report_scalar("epoch time (seconds)", "total", value=one_epoch_time, iteration=epoch)
            log_segmentation_metrics(logger, "val", confmat, dice, epoch, class_names)
        one_epoch_time = str(datetime.timedelta(seconds=int(one_epoch_time)))
        print(f"training epoch {epoch} time {one_epoch_time}")
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.8f}\n" \
                         f"val_dice: {dice:.3f}\n" \
                         f"epoch time: {one_epoch_time}\n"

            f.write(train_info + val_info + "\n\n")

        if not args.save_best:
            torch.save(model.state_dict(), OUTPUT_SAVE_PATH / f"{model_name}_epoch{epoch}_dice{dice:.3f}.pth")

        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
            best_model_path = OUTPUT_SAVE_PATH / f"{model_name}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            with open(OUTPUT_SAVE_PATH / f"{model_name}_best.txt", "w") as f:
                f.write(train_info + val_info)
            if best_output_model is not None:
                # Reuse one model entry; finish uploading before this file is overwritten.
                best_output_model.update_weights(
                    weights_filename=str(best_model_path), auto_delete_file=False,
                    iteration=epoch, async_enable=False)
                best_output_model.report_scalar("dice", "val", value=dice, iteration=epoch)
                logger.report_single_value("best_val_epoch", epoch)

        if logger is not None:
            logger.report_scalar("dice", "best_val", value=best_dice, iteration=epoch)
            logger.report_single_value("best_val_dice", best_dice)
        save_checkpoint(
            save_path=CHECKPOINT_FILE, epoch=epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
            scaler=scaler, best_dice=best_dice, train_loss=train_loss, dice_coefficient=dice_coefficient,
            best_epoch=best_epoch, best_model_path=best_model_path)

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
                        default=r"G:\Devendra\ASPHALT\TRAININGNEW\SPLIT",
                        help="root")
    parser.add_argument("--num-classes", default=5, type=int)  # exclude background
    parser.add_argument("--aux", default=True, type=bool, help="deeplabv3 auxilier loss")
    parser.add_argument("--phi", default="b0", help="Use backbone")
    parser.add_argument('--pretrained', default=False, type=bool, help='backbone')
    parser.add_argument('--pretrained-weights', type=str,
                        default=r"",
                        help='pretrained weights path')
    parser.add_argument('--optimizer-type', default="adamw")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')  # 0.00006
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')

    parser.add_argument('--save-best', default=False, type=bool, help='only save best dice weights')
    parser.add_argument('--resume', default=str(CHECKPOINT_FILE), help='resume from checkpoint')
    parser.add_argument('--no-clearml', action='store_true', help='disable ClearML tracking')
    parser.add_argument('--clearml-project', default='c3d-Segmentation')
    parser.add_argument('--clearml-task-name', default=model_name)
    parser.add_argument('--clearml-output-uri', default='', help='model upload destination; defaults to ClearML file server')
    parser.add_argument('--clearml-dataset-project', default='',
                        help='ClearML dataset project; defaults to --clearml-project')
    parser.add_argument('--clearml-dataset-name', default='Asphalt',
                        help='ClearML dataset name to download before training')
    parser.add_argument('--clearml-dataset-version', default='',
                        help='Optional ClearML dataset version')
    parser.add_argument('--clearml-dataset-id', default='',
                        help='Optional ClearML dataset id/hash; overrides project/name/version')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for automatic mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    task = None
    if not args.no_clearml:

        task = Task.init(
            project_name=args.clearml_project, task_name=args.clearml_task_name,
            output_uri=args.clearml_output_uri or True,
            auto_connect_frameworks={"pytorch": False},
            auto_connect_arg_parser=False, reuse_last_task_id=False)
        task.connect(args, name="Args")
    try:
        main(args, task=task)
    except Exception as exc:
        if task is not None:
            task.mark_failed(status_reason=str(exc))
        raise
    finally:
        if task is not None:
            task.close()
