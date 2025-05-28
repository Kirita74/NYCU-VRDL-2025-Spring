import os
import argparse
import torch
import numpy as np
import gc
from tqdm import tqdm
from PIL import Image, ImageFilter
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from dataset import getDataset, TestDataset
from PromptIR import PromptIR
from utils import save_checkpoint, load_checkpoint, Charbonnier_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.225, 0.225]).view(1, 3, 1, 1)

torch.backends.cudnn.benchmark = True

psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
msssim = MultiScaleStructuralSimilarityIndexMeasure(
    data_range=1.0, betas=(0.0448, 0.2856, 0.3001, 0.2363)).to(DEVICE)


def unnormalize(x: torch.Tensor):
    mean = _IMAGENET_MEAN.to(x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = _IMAGENET_STD.to(x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def ssim_calc(img1: torch.Tensor, img2: torch.Tensor):
    return ssim(unnormalize(img1), unnormalize(img2))


def psnr_calc(img1: torch.Tensor, img2: torch.Tensor):
    return psnr(unnormalize(img1), unnormalize(img2))


def msssim_loss(img1: torch.Tensor, img2: torch.Tensor):
    return 1 - msssim(unnormalize(img1), unnormalize(img2))


def loadData():
    train_dataset, valid_dataset = getDataset(TRAIN_PATH)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, valid_dataloader


def train():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    log_dir = os.path.join("logs", LOG_DIR)
    writer = SummaryWriter(log_dir=log_dir)
    train_dataloader, valid_dataloader = loadData()

    model = PromptIR(inp_channels=3, decoder=True)
    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = Charbonnier_loss()
    scaler = GradScaler()
    loss_coef = 0.842
    start_epoch = 0

    swa_model = AveragedModel(model, device=DEVICE)
    swa_scheduler = SWALR(optimizer=optimizer, swa_lr=1e-6)
    swa_start = (start_epoch + EPOCHS) * 0.8

    if (PRETRAIN_WEIGHT_PATH != None):
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, PRETRAIN_WEIGHT_PATH, optimizer, scheduler)
    best_val_loss = float('inf')
    best_val_PSNR = 0.0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # Train
        model.train()
        gc.collect()
        epoch_loss = 0.0
        for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            if (DEVICE == torch.device("cuda")):
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda", cache_enabled=True):
                restore = model(images)
                if (epoch < 30):
                    loss = loss_fn(restore, targets)
                else:
                    loss = (1-loss_coef) * loss_fn(restore, targets) + \
                        loss_coef * msssim_loss(restore, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item()

        if (epoch >= swa_start):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        avg_loss = epoch_loss/len(train_dataloader)
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/Epoch_loss", avg_loss, epoch)

        # Valid
        model.eval()
        psnr_sum = 0.0
        ssim_sum = 0.0
        valid_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_dataloader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                with autocast(device_type="cuda", cache_enabled=True):
                    restore = model(images)
                    valid_loss += (1-loss_coef) * loss_fn(restore, targets) + \
                        loss_coef * msssim_loss(restore, targets)
                    psnr_sum += psnr_calc(restore, targets)
                    ssim_sum += ssim_calc(restore, targets)

        avg_loss = valid_loss / len(valid_dataloader)
        avg_psnr = psnr_sum / len(valid_dataloader)
        avg_ssim = ssim_sum / len(valid_dataloader)
        print(
            f"Valid: AVG_LOSS: {avg_loss:.4f}\tAVG_PSNR: {avg_psnr:.4f}\tAVG_SSIM: {avg_ssim:.4f}")
        writer.add_scalar("Valid/Loss", avg_loss, epoch)
        writer.add_scalar("Valid/PSNR", avg_psnr, epoch)
        writer.add_scalar("Validk/SSIM", avg_ssim, epoch)

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_PSNR": best_val_PSNR
        }
        save_checkpoint(state, is_best=avg_loss <
                        best_val_loss, ckpt_dir="checkpoints", fileDir=SAVE_DIR)

    update_bn(train_dataloader, swa_model, device=DEVICE)
    torch.save(swa_model.module.state_dict(),
               f'{SAVE_DIR}\\swa_best-model_weight.pth')


def test():
    os.makedirs("test_img", exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.225, 0.225], device='cuda').view(1, 3, 1, 1)
    test_dataset = TestDataset(TEST_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = PromptIR(decoder=True)
    model, _, _, _ = load_checkpoint(
        model, ckpt_path=PRETRAIN_WEIGHT_PATH)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for filenames, images in tqdm(test_dataloader, desc="Testing..."):
            images = images.to(DEVICE)

            output = model(images)
            output = (output * std + mean) * 255.0
            output = output.cpu().detach().numpy()
            output = np.clip(output, 0.0, 255.0).astype(
                np.uint8).squeeze(axis=0)
            image = output.transpose((1, 2, 0))
            image = Image.fromarray(image, mode="RGB")
            image.save(f'test_img\\{filenames[0]}')


def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        help="Mode of train or test"
    )
    parser.add_argument(
        "DATAPATH",
        type=str,
        default="..\\hw4_realse_dataset",
        help="Root directory of the dataset")
    parser.add_argument(
        "SAVE_DIR",
        type=str,
        default="weight",
        help="Path to save model weight")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate")
    parser.add_argument(
        "--decay",
        type=float,
        default=5e-5,
        help="Weight decay for optimizer")
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=None,
        help="Path of pretrained weight")
    parser.add_argument(
        '--log_dir',
        type=str,
        default="logs/log1",
        help="Folder of training log")
    return parser


if __name__ == "__main__":
    args = argParse()

    TRAIN_PATH = os.path.join(args.DATAPATH, "train")
    TEST_PATH = os.path.join(args.DATAPATH, "test")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    OPTIMIZER_WEIGHT_DECAY = args.optimizer_weight_decay
    SAVE_DIR = args.SAVE_DIR
    LOG_DIR = args.lod_dir
    PRETRAIN_WEIGHT_PATH = args.pretrained_weight_path
    if (args.mode == "train"):
        train()
    elif (args.mode == "test"):
        test()
