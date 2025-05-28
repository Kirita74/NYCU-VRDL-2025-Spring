import os
import torch
import random
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms.v2 import functional as F


def applyCubblur(img1: Image, img2: Image, alpha=0.5):
    if random.random() > 0.5:
        return img1, img2

    W, H = img1.size
    cut_ratio = random.uniform(0.3, alpha)

    cH = int(H * cut_ratio)
    cW = int(W * cut_ratio)

    cy = random.randint(0, H - cH)
    cx = random.randint(0, W - cW)

    crop = img2.crop((cx, cy, cx + cW, cy + cH))
    img1.paste(crop, (cx, cy))

    return img1, img2


def applyTransform(img1, img2, isTrain: bool):

    if (isTrain == False):
        transform = T.Compose([
            # T.Resize((128, 128)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.225, 0.225])
        ])
        if (img2 == None):
            return transform(img1)
        return transform(img1), transform(img2)
    else:
        # i, j, h, w = T.RandomCrop.get_params(img1, output_size=(128, 128))
        # img1 = F.crop(img1, i, j, h, w)
        # img2 = F.crop(img2, i, j, h, w)

        # img1, img2 = applyCubblur(img1, img2)
        if random.random() < 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        if random.random() < 0.5:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)

        to_tensor = T.Compose([T.ToImage(),
                               T.ToDtype(torch.float32, scale=True),
                               T.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.225, 0.225])])
        return to_tensor(img1), to_tensor(img2)


class CustomDataset(Dataset):
    def __init__(self, train_path: str, mode):
        super().__init__()
        self.degrade_root = os.path.join(train_path, "degraded")
        self.degraded_list = sorted(os.listdir(self.degrade_root)[1:])
        self.clean_root = os.path.join(train_path, "clean")
        self.isTrain = (mode == "train")

    def __getitem__(self, index):
        degraded_img_path = os.path.join(
            self.degrade_root, self.degraded_list[index])
        clean_img_filename = self.degraded_list[index].replace("-", "_clean-")
        clean_img_path = os.path.join(self.clean_root, clean_img_filename)

        degraded_img = Image.open(degraded_img_path).convert('RGB')
        clean_img = Image.open(clean_img_path,).convert('RGB')

        d_tensor, c_tensor = applyTransform(
            degraded_img, clean_img, self.isTrain)
        return d_tensor, c_tensor

    def __len__(self):
        return len(self.degraded_list)


def getDataset(train_path, scale=0.85):
    full_train_ds = CustomDataset(train_path, "train")
    full_valid_ds = CustomDataset(train_path, "valid")

    n = len(full_train_ds)
    idx = list(range(n))
    train_n = int(scale * n)
    train_idx, valid_idx = idx[:train_n], idx[train_n:]

    train_ds = Subset(full_train_ds, train_idx)
    valid_ds = Subset(full_valid_ds, valid_idx)

    return train_ds, valid_ds


class TestDataset:
    def __init__(self, test_path: str):
        super().__init__()
        self.degrade_root = os.path.join(test_path, "degraded")
        self.degraded_list = sorted(os.listdir(self.degrade_root))
        self.tranform = T.Compose([T.ToImage(),
                                   T.ToDtype(torch.float32, scale=True),
                                   T.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.225, 0.225])])

    def __getitem__(self, index):
        degraded_img_path = os.path.join(
            self.degrade_root, self.degraded_list[index])
        degraded_img = Image.open(degraded_img_path).convert('RGB')
        degraded_img = self.tranform(degraded_img)

        return self.degraded_list[index], degraded_img

    def __len__(self):
        return len(self.degraded_list)
