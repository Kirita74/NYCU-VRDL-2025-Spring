import os
import gc
import torch
import json
import argparse
from tqdm import tqdm
from dataset import TestDataset, getDatasets
from torch.utils.data.dataloader import DataLoader
from model import CustomedModel
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
from utils import encode_mask
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

anchor_generator = AnchorGenerator(
    sizes=((4,), (8,), (16,), (32,), (64,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)

mask_roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=14,
    sampling_ratio=4
)


def load_data():
    train_dataset, valid_dataset = getDatasets(train_path=TRAIN_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader, valid_loader


def train():
    tensorboard_writer = SummaryWriter(LOG_DIR)
    train_loader, valid_loader = load_data()
    model = CustomedModel(
        num_classes=NUM_CLASSES + 1,
        anchor_generator=anchor_generator,
        roi_pooler=roi_pooler,
        mask_roi_pooler=mask_roi_pooler,
        pretrained=True
    )

    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(device='cuda')
    val_metric = MeanAveragePrecision()
    best_map = 0.0

    for epoch in range(EPOCHS):
        model.train()
        gc.collect()
        epoch_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images_gpu = [img.to(DEVICE) for img in images]
            targets_gpu = [{k: v.to(DEVICE)
                            for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(device_type='cuda', cache_enabled=True):
                loss_dict = model(images_gpu, targets_gpu)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Loss: {avg_epoch_loss:.4f}")

        tensorboard_writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        tensorboard_writer.add_scalar(
            "Loss/Learning rate", scheduler.get_last_lr()[0], epoch
        )

        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]

                predicteds = model(images)
                predicteds = [{k: v.cpu() for k, v in p.items()}
                              for p in predicteds]

                processed_pred = []
                for pred in predicteds:
                    mask = torch.sigmoid(pred["masks"])
                    mask = mask.detach().numpy()
                    mask = mask > 0.5

                    if mask.sum() == 0:
                        continue

                    processed_pred.append(pred)

                if len(processed_pred) == 0:
                    continue
                val_metric.update(preds=processed_pred, target=targets)

        metrics = val_metric.compute()
        print(
            f"Valid: [Epoch {epoch + 1}] mAP: {metrics['map']:.4f}, "
            f"mAP50: {metrics['map_50']:.4f}, mAP75: {metrics['map_75']:.4f}"
        )
        tensorboard_writer.add_scalar('mAP/val', metrics['map'], epoch)
        tensorboard_writer.add_scalar('mAP50/val', metrics['map_50'], epoch)
        tensorboard_writer.add_scalar("mAP75/val", metrics['map_75'], epoch)

        with open(f"{LOG_DIR}\\mAP_record.txt", "a+") as map_writer:
            map_writer.writelines(f"[Epoch {epoch + 1}]\n")
            for k, v in metrics.items():
                if k == "classes":
                    continue
                if isinstance(v, torch.Tensor):
                    v = float(v)
                map_writer.writelines(f"{k}:{v:.4f}\n")

        if metrics["map"] > best_map:
            torch.save(model.state_dict(), PRETRAINED_WEIGHT_PATH)


def test():
    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CustomedModel(
        num_classes=NUM_CLASSES + 1,
        anchor_generator=anchor_generator,
        roi_pooler=roi_pooler,
        mask_roi_pooler=mask_roi_pooler,
        pretrained=False
    )
    model.load_pretrained_weight(PRETRAINED_WEIGHT_PATH)
    model.to(DEVICE)
    model.eval()

    results = []

    for file_id, file_size, images in tqdm(test_loader, desc="Testing"):
        file_id = file_id.item()
        images = [img.to(DEVICE) for img in images]
        predicted = model(images)
        predicted = [{k: v.cpu() for k, v in p.items()} for p in predicted]
        for p in predicted:
            bboxes = p["boxes"]
            labels = p["labels"]
            scores = p["scores"]
            masks = p["masks"]

            for idx in range(bboxes.shape[0]):
                bbox = bboxes[idx].tolist()
                label = labels[idx].item()
                score = scores[idx].item()
                mask = torch.sigmoid(masks[idx])
                mask = mask.detach().numpy()
                mask = mask > 0.5

                if mask.sum() == 0:
                    continue

                x_min, y_min, x_max, y_max = bbox
                if x_min >= file_size[1] or y_min >= file_size[0]:
                    continue

                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                rle_mask = encode_mask(mask.squeeze(axis=0))
                result = {
                    "image_id": file_id,
                    "bbox": bbox,
                    "score": score,
                    "category_id": label,
                    "segmentation": rle_mask
                }
                results.append(result)

    with open("test-results.json", "w", encoding='utf-8') as outfile:
        json.dump(results, outfile)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode",
                        type=str, default="traing",
                        help="Mode of training or testing")
    parser.add_argument(
        "DATAPATH",
        type=str,
        default="..\\data",
        help="Root directory of the dataset")
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
        "--optimizer_weight_decay",
        type=float,
        default=5e-5,
        help="Weight decay for AdamW")
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=None,
        help="Path of pretrained weight")
    parser.add_argument(
        "--save_path",
        type=str,
        default="weight.pth",
        help="Path to save model weight")
    parser.add_argument(
        '--log_dir',
        type=str,
        default="logs/log1",
        help="Folder of training log")
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Mask threshold(0.0 ~ 1.0)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    TRAIN_PATH = os.path.join(args.DATAPATH, "train")
    TEST_PATH = os.path.joins(args.DATAPATH, "test-release")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    OPTIMIZER_WEIGHT_DECAY = args.optimizer_weight_decay
    LOG_DIR = args.log_dir
    PRETRAINED_WEIGHT_PATH = args.pretrained_weight_path
    MASK_THRESHOLD = args.mask_threshold
    MODE = args.mode

    if (MODE == "train"):
        train()
    elif (MODE == "test"):
        test()
    else:
        print("Invalid mode, please use 'train' or 'test'")
