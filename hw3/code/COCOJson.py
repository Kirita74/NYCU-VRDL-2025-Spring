import os
import re
import json
import argparse
import numpy as np
import skimage.io as sio
from scipy import ndimage
from utils import encode_mask

annotation_infos = []
annotation_id = 1


def mask_instance(image_id, mask_label, mask_path):
    """
    Process a mask instance and generate annotation information.

    """
    global annotation_id
    mask = sio.imread(mask_path)
    binary_mask = mask > 0

    ys, xs = np.where(binary_mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    labeled, num_objs = ndimage.label(binary_mask)

    for i in range(1, num_objs + 1):  # labeled == 0 will output all mask instances
        mask_instance = labeled == i

        # Bounding box
        ys, xs = np.where(mask_instance)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        if xmax <= xmin or ymax <= ymin:
            continue

        anno_info = {
            "image_id": image_id,
            "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
            "category_id": mask_label,
            "segmentation": encode_mask(mask_instance),
        }

        annotation_infos.append(anno_info)
        annotation_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "DATAPATH",
        type=str,
        help="Root directory of the dataset"
    )

    args = parser.parse_args()
    train_path = os.path.join(args.DATAPATH, "train")
    dirs = os.listdir(train_path)

    img_infos = []

    for idx, subdir in enumerate(dirs):
        image_id = idx + 1

        for filename in os.listdir(os.path.join(train_path, subdir)):
            if filename == "image.tif":
                image_path = os.path.join(train_path, subdir, filename)
            else:
                mask_label = int(re.findall(r"\d+", filename)[0])
                mask_instance(
                    image_id,
                    mask_label,
                    os.path.join(train_path, subdir, filename),
                )

        img_info = {
            "id": image_id,
            "filename": subdir,
        }
        img_infos.append(img_info)

    categories = [
        {"id": i + 1, "name": i + 11} for i in range(4)
    ]

    json_dict = {
        "images": img_infos,
        "annotations": annotation_infos,
        "categories": categories,
    }

    with open(os.path.join(train_path, "sample.json"), "w") as outfile:
        json.dump(json_dict, outfile)
