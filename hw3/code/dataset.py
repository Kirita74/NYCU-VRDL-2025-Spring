import os
import json
import numpy as np
import cv2
import torch
from pycocotools import mask as mask
from utils import decode_maskobj
import albumentations as albu
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomedDataset(Dataset):
    def __init__(self, file_path, transform):
        super().__init__()
        json_data = json.load(open(f"{file_path}\\sample.json", "r"))
        self.images = json_data["images"]
        self.annotations = json_data["annotations"]
        self.categories = json_data["categories"]
        self.folder = file_path
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.images[idx]["filename"]

        img = cv2.imread(os.path.join(self.folder, filename, "image.tif"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        file_id = self.images[idx]["id"]

        image_annotations = [
            anno for anno in self.annotations if anno["image_id"] == file_id
        ]

        boxes = []
        labels = []
        binary_masks = []

        for anno in image_annotations:
            x, y, bw, bh = anno["bbox"]
            x_min = x
            y_min = y
            x_max = x + bw
            y_max = y + bh

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(anno["category_id"]))
            binary_masks.append(
                decode_maskobj(anno["segmentation"]).astype(np.uint8)
            )

        augmented = self.transform(
            image=img, bbox=boxes, masks=binary_masks, labels=labels
        )
        image = augmented["image"]
        boxes = augmented["bbox"]
        labels = augmented["labels"]
        binary_masks = augmented["masks"]
        binary_masks_np = np.array(binary_masks)

        target = {
            "image_id": torch.tensor([file_id]),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(binary_masks_np),
        }
        return image, target

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(self, test_path):
        super().__init__()
        self.root = test_path
        with open("..\\data\\test_image_name_to_ids.json") as rdr:
            self.json_loader = json.load(rdr)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.225, 0.225])
        ])

    def __getitem__(self, index):
        json_data = self.json_loader[index]
        filename = json_data["file_name"]
        file_id = json_data["id"]
        h = json_data['height']
        w = json_data['width']

        filepath = os.path.join(self.root, filename)
        img = Image.open(filepath).convert("RGB")
        img_transformed = self.transform(img)
        file_size = (w, h)
        return file_id, file_size, img_transformed

    def __len__(self):
        return len(self.json_loader)


def get_datasets(train_path, seed=42):
    bbox_params = albu.BboxParams(format="pascla_voc", label_fields=['labels'])
    train_transform = albu.Compose([
        albu.HorizontalFlip(p=0.3),
        albu.VerticalFlip(p=0.3),
        albu.GaussNoise(p=0.2),
        albu.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.225, 0.225]),
        albu.ToTensorV2()
    ], bbox_params=bbox_params)

    valid_transform = albu.Compose([
        albu.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.225, 0.225]),
        albu.ToTensorV2()
    ], bbox_params=bbox_params)

    full_train_ds = CustomedDataset(train_path, transform=train_transform)
    full_valid_ds = CustomedDataset(train_path, transform=valid_transform)

    import random
    from torch.utils.data import Subset

    n = len(full_train_ds)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    train_n = int(0.9 * n)
    train_idx, valid_idx = idx[:train_n], idx[train_n:]

    train_ds = Subset(full_train_ds, train_idx)
    valid_ds = Subset(full_valid_ds, valid_idx)

    return train_ds, valid_ds