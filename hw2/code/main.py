import torch
import json
import csv
import os
import argparse
import random
from torchvision.utils import to_pil_image
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from matplotlib import patches
from dataset import CustomDataset
from tqdm import tqdm
from PIL import Image
from model import CustomModel
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.rpn import AnchorGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

anchoer_generator = AnchorGenerator(
    sizes=((4,), (8,), (16,), (32,), (64,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)


def get_transform():
    """
    Get data augments for training
    """
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2),
        # transforms.GaussianBlur(kernel_size=[3,3]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.225, 0.225]),
    ])

    return transform


def load_data():
    """
    Load data
    """
    train_json_file = open(f"{TRAIN_DATA_PATH}.json", mode='r')
    train_json_data = json.load(train_json_file)
    categories = train_json_data["categories"]

    transform = get_transform()
    train_dataset = CustomDataset(f"{TRAIN_DATA_PATH}.json", transform)
    valid_dataset = CustomDataset(f"{VAL_DATA_PATH}.json")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, valid_loader, valid_dataset, len(categories)


def train():
    """
    Training
    """
    writer = SummaryWriter(log_dir=LOG_PATH)

    train_loader, valid_loader, valid_dataset, num_categories = load_data()
    model = CustomModel(num_categories + 1, anchor_generator=anchoer_generator,
                        roi_pooler=roi_pooler, pretrained=True)
    # model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)

    num_epochs = EPOCHS

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    val_metric = MeanAveragePrecision()
    best_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # transfer data to GPU
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            # update weight
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()
        scheduler.step()
        avg_loss = epoch_loss/len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        model.eval()
        val_metric.reset()

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()}
                           for t in targets]

                predicteds = model(images)
                predicteds = [{k: v.cpu() for k, v in p.items()}
                              for p in predicteds]

                for pred, target in zip(predicteds, targets):
                    print("Predicted boxes:", pred['boxes'].shape)
                    print("Predicted labels:", pred['labels'])
                    print("Target boxes:", target['boxes'].shape)
                    print("Target labels:", target['labels'])
                val_metric.update(preds=predicteds, target=targets)
        metrics = val_metric.compute()
        print(
            f"Valid: [Epoch {epoch+1}] mAP: {metrics['map']:.4f}, mAP50: {metrics['map_50']:.4f} mAP75: {metrics['map_75']:.4f}")
        writer.add_scalar('mAP/val', metrics['map'], epoch)
        writer.add_scalar('mAP50/val', metrics['map_50'], epoch)
        writer.add_scalar("mAP75/val", metrics['map_75'], epoch)

        if metrics["map"] > best_map:
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)

        label_map = {i: str(i - 1) for i in range(1, 11)}

        show_random_prediction(
            model=model,
            dataset=valid_dataset,
            device=DEVICE,
            label_map=label_map,
            score_thresh=0.7,
        )

    writer.close()


TEST_PATH = 'data/test'


def test():
    """
    Test the model
    """
    test_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225]),
    ])
    imags_paths = [os.path.join(TEST_PATH, img) for img in os.listdir(
        TEST_PATH) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    model = CustomModel(11, anchor_generator=anchoer_generator,
                        roi_pooler=roi_pooler, pretrained=True)
    model.base_model.roi_heads.score_thresh = 0.7

    model.load_state_dict(torch.load(
        "resnet101_v2_model_1.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    results = []

    for img_path in imags_paths:
        img = Image.open(img_path).convert('RGB')
        img_id = os.path.basename(img_path).split(sep='.')[0]
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predicteds = model(img_tensor)

            for pred in predicteds:
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                for bbox, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1

                    results.append({
                        'image_id': int(img_id),
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': float(score),
                        'category_id': int(label)
                    })

    output_path = "pred.json"
    with open(output_path, 'w') as f:
        json.dump(results, f)


def generate_task2(json_path='pred.json', output_path='pred.csv'):
    """
    Generate task 2 submission file
    """
    with open(json_path, 'r') as f:
        detections = json.load(f)

    results_by_image = {}
    for detection in detections:
        image_id = detection['image_id']
        if image_id not in results_by_image:
            results_by_image[image_id] = []

        results_by_image[image_id].append({
            'box': detection['bbox'],
            'score': detection['score'],
            'category_id': detection['category_id']
        })

    results = []
    for image_id, detections in results_by_image.items():
        if not detections:
            pred_label = -1
        else:
            detections.sort(key=lambda x: x['box'][0])

            digit_values = [str(int(d['category_id'])-1) for d in detections]
            try:
                pred_label = int(''.join(digit_values))
            except ValueError:
                pred_label = -1

        results.append([int(image_id), pred_label])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'pred_label'])
        writer.writerows(results)

    return results


def show_random_prediction(model, dataset, device, label_map=None, score_thresh=0.7):
    """
    Show and save a random prediction from the dataset using the model
    """

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(-1, 1, 1)

    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    image, target = dataset[idx]

    with torch.no_grad():
        pred = model([image.to(DEVICE)])[0]

    denorm_image = image.to(device) * std + mean
    denorm_image = denorm_image.clamp(0, 1)

    denorm_image = to_pil_image(denorm_image.cpu())
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(denorm_image)

    # predict
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.tolist()
        class_name = label_map[label.item()] if label_map else str(
            label.item())
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(x1, y1 - 5, f"{class_name} ({score:.2f})",
                color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    # target
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        class_name = label_map[label.item()] if label_map else str(
            label.item())
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'))
        ax.text(x1, y1 + 10, f"GT: {class_name}",
                color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')
    plt.tightlayout()

    plt.savefig(f"predcit image\img{idx}.jpg",
                bbox_inches='tight', padinches=0.1)
    print(f"image saved:img{idx}.jpg")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode",
                        type=str, default="traing",
                        help="Mode of training or testing")
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
        default=1e-3,
        help="Learning rate")
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW")
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
        default="logs/log1",
        help="Folder of training log")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    TRAIN_DATA_PATH = os.path.join(args.DATAPATH, "train")
    VAL_DATA_PATH = os.path.join(args.DATAPATH, "val")
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    OPTIMIZER_WEIGHT_DECAY = args.optimizer_weight_decay
    WEIGHT_PATH = args.pretrained_weight_path
    WEIGHT_SAVE_PATH = args.save_path
    LOG_PATH = args.log_dir
    MODE = args.mode

    if (MODE == "train"):
        train()
    elif (MODE == "test"):
        test()
        generate_task2()
    else:
        print("Invalid mode, please use 'train' or 'test'")
