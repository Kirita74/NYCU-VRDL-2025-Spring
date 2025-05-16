import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, filePath: str, transform=None):
        super().__init__()
        json_data = json.load(open(filePath, "r"))
        self.images = json_data["images"]
        self.annotations = json_data["annotations"]
        self.categories = json_data["categories"]
        self.folder = filePath.split(sep='.')[0]
        if (transform == None):
            self.transforms = transforms.Compose([

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.225, 0.225]),
            ])
        else:
            self.transforms = transform

    def __getitem__(self, index):
        filename = self.images[index]["file_name"]

        img = Image.open(f"{self.folder}\\{filename}").convert("RGB")

        file_id = self.images[index]["id"]

        image_annotations = [
            anno for anno in self.annotations if anno["image_id"] == file_id]
        # image_categories = [self.categories[anno["category_id"]-1]["name"] for anno in image_annotations]

        # h,w,_ = img.shape
        boxes = []
        labels = []

        for anno in image_annotations:
            x, y, bw, bh = anno["bbox"]
            boxes.append([x, y, x+bw, y+bh])
            labels.append(anno["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img = self.transforms(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([file_id])
        }

        return img, target

    def __len__(self):
        return len(self.images)
