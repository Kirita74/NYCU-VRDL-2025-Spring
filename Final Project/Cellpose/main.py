"""
# Cellpose Training Script
This script is designed to train, validate, or test a Cellpose model for image segmentation tasks.
"""

import os
import argparse
import multiprocessing

import torch
from torch.backends import cudnn
from torch.optim import AdamW

# third-party imports
from cellpose import models

# local imports
from train import train_model, load_data, validate_model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main function for handling train, validate, and test modes."""
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Cellpose training script for segmentation tasks.")

    parser.add_argument("--data_path", default="", type=str, help="Root path to dataset.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Epochs.")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument(
        "-d", "--decay", type=float, default=0.1, help="Weight decay (for AdamW).")
    parser.add_argument(
        "-s", "--saved_path", type=str, default="./saved_models_fold5_v2",
        help="Path to save models and logs.")
    parser.add_argument(
        "-im", "--image_size", type=int, default=256, help="Input image size for model.")
    parser.add_argument(
        "-m", "--mode", type=str, default="train", choices=["train", "validate", "test"],
        help="Execution mode.")

    args = parser.parse_args()

    os.makedirs(args.saved_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    train_loader, valid_loader = load_data(
        train_data_dir='Cellpose_5fold_train/fold5',
        valid_data_dir='Cellpose_5fold_valid/fold5',
        args=args)

    model = models.CellposeModel(
        gpu=True,
        pretrained_model="saved_models_fold5/best_model_0.3117264211177826.pth"
    )

    print(
        f"Model has {count_parameters(model.net) / 1e6:.2f}M trainable parameters")

    optimizer = AdamW(
        model.net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay
    )

    if args.mode == 'train':
        train_model(device=device, model=model, optimizer=optimizer,
                    train_loader=train_loader, valid_loader=valid_loader, args=args)
    elif args.mode == 'validate':
        validate_model(device=device, model=model, optimizer=optimizer,
                       valid_loader=valid_loader, args=args)


if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()
        torch.multiprocessing.set_start_method('spawn', force=True)
    else:
        torch.multiprocessing.set_start_method('spawn', force=True)

    main()
