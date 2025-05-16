import os
import json
import argparse
import csv
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from model import CustomResnextModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.225, 0.225]
        )
    ])

    return transform


def get_csv_writer():
    csvfile = open(CSV_FILE_PATH, 'w+', newline='')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['image_name', 'pred_label'])

    return csvwriter


def test():
    test_transform = get_transform()

    with open('class_mapping.json', 'r') as r:
        class_to_idx = json.load(r)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    imags_paths = [os.path.join(TEST_PATH, img) for img in os.listdir(
        TEST_PATH) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    model = CustomResnextModel(num_classes=len(class_to_idx), pretrained=False)
    model.load_weight(PRETRAINED_WEIGHT_PATH)
    model.to(device=DEVICE)
    model.eval()

    csvwriter = get_csv_writer()

    for img_path in imags_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            _, predict_label = torch.max(output, 1)
        img_name = os.path.split(img_path)[-1].split(sep='.')[0]
        predict_label = idx_to_class[predict_label.item()]
        csvwriter.writerow([img_name, predict_label])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "test_data_path",
        type=str,
        default="data/test",
        help="Path of test data")
    parser.add_argument(
        "pretrained_weight_path",
        type=str,
        help="Path of pretrained weight")
    parser.add_argument(
        "--prediction_csv_path",
        type=str,
        default="prediction.csv",
        help="Path of prediction csv")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    TEST_PATH = args.test_data_path
    PRETRAINED_WEIGHT_PATH = args.pretrained_weight_path
    CSV_FILE_PATH = args.prediction_csv_path

    test()