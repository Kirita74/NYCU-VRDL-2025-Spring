import gc
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import CustomResnextModel


CLASS_MAPPING_FILE = "class_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """
    Get the data augmentations
    """

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.225, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.225, 0.225]
        )
    ])

    return train_transform, val_transform


def load_data():
    """
    Load data from train and val dataset
    """

    train_transform, val_transform = get_transforms()

    train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_DATA_PATH,
        transform=val_transform
    )

    with open(CLASS_MAPPING_FILE, "w+") as f:
        json.dump(train_dataset.class_to_idx, f)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train():
    '''
    Training model
    '''
    writer = SummaryWriter(log_dir=LOG_PATH)

    train_dataset, val_dataset, train_dataloader, val_dataloader = load_data()
    num_classes = len(train_dataset.classes)

    if (WEIGHT_PATH is None):
        model = CustomResnextModel(num_classes=num_classes, pretrained=True)
    else:
        model = CustomResnextModel(num_classes=num_classes, pretrained=False)
        model.load_weight(path=WEIGHT_PATH)
    model.to(device=DEVICE)

    # Setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=ETA_MIN)

    best_valid_acc = 0.0

    for epoch in range(EPOCHS):
        gc.collect()
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        model.train()
        running_loss = 0.0
        losses = []

        training_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            total=len(train_dataloader)
        )

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, target_a, target_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(loss_fn, outputs, target_a, target_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            training_bar.update(1)
            training_bar.set_postfix(
                loss=f"{running_loss / len(train_dataloader):.4f}")

            writer.add_scalar(
                "Training Loss (Batch)",
                loss.item(),
                epoch * len(train_dataloader) + batch_idx
            )

        training_loss = np.mean(losses)
        writer.add_scalar("Training Loss (Epoch)", training_loss, epoch)

        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0.0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels).item()
                valid_loss += loss
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_acc = 100 * valid_correct / valid_total
        valid_loss /= valid_total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"train Loss: {training_loss:.4f} - "
            f"validation accuracy: {valid_acc:.2f}%"
        )

        writer.add_scalar("Validation Accuracy", valid_acc, epoch)
        writer.add_scalar("Validation Loss", valid_loss, epoch)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model.save_weight(path=WEIGHT_SAVE_PATH)

        scheduler.step(valid_acc)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "DATAPATH",
        type=str,
        default="./data",
        help="Root directory of the dataset")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate")
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW")
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Minimum learing rate for scheduler")
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=None,
        help="Path of pretrained weight")
    parser.add_argument(
        "--save_path",
        type=str,
        default="weightp.pth",
        help="Path to save model weight")
    parser.add_argument(
        '--log_dir',
        type=str,
        default="logs",
        help="Folder of training log")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    TRAIN_DATA_PATH = os.path.join(args.DATAPATH, "train")
    VAL_DATA_PATH = os.path.join(args.DATAPATH, "val")
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    OPTIMIZER_WEIGHT_DECAY = args.optimizer_weight_decay
    ETA_MIN = args.eta_min
    WEIGHT_PATH = args.pretrained_weight_path
    WEIGHT_SAVE_PATH = args.save_path
    LOG_PATH = args.log_dir

    train()