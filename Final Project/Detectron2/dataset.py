"""
This script is designed to train a Detectron2 model 
    for instance segmentation on the Sartorius Cell dataset.
It uses a custom evaluator to compute the mean Average Precision (MaP) 
    based on Intersection over Union (IoU) for the model's predictions.
"""

import warnings
from pathlib import Path

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Ignore "future" warnings and Data-Frame-Slicing warnings.
warnings.filterwarnings('ignore')


# detectron2


def register_datasets():
    """
    Register datasets for Detectron2 training and evaluation.
    This function registers multiple datasets, including a multi-class dataset
    and several one-class datasets, each with its own training and validation splits.
    """

    data_dir = Path('sartorius')
    patched_dir = Path("sartorius_patched")

    # 多類資料集
    DatasetCatalog.clear()
    # register_coco_instances(
    #     "sartorius_Cell_train",
    #     {},
    #     "crossvalidationfold5/coco_cell_train_fold3.json",
    #     str(data_dir)
    # )
    register_coco_instances(
        "sartorius_Cell_train",
        {},
        "sartorius_patched_5fold/annotations_train_patched_fold1.json",
        "sartorius_patched_5fold/train_images_fold1"
    )

    register_coco_instances(
        "sartorius_Cell_valid",
        {},
        "crossvalidationfold5/coco_cell_valid_fold1.json",
        str(data_dir)
    )

    # 以下是 1-class 資料集（假設你已經分好 coco json）
    one_class_sets = ["shsy5y", "astro", "cort"]
    for cls in one_class_sets:
        train_name = f"{cls}_train"
        val_name = f"{cls}_valid"
        train_json = patched_dir / f"annotations_train_patched_{cls}.json"
        val_json = patched_dir / f"annotations_val_{cls}.json"
        train_img_dir = patched_dir / f"train_images_{cls}"
        val_img_dir = patched_dir / f"val_images_{cls}"

        register_coco_instances(
            train_name,
            {},
            str(train_json),
            str(train_img_dir))
        register_coco_instances(val_name, {}, str(val_json), str(val_img_dir))

    # 測試是否註冊成功（可以刪掉）
    print("Datasets registered:")
    print(DatasetCatalog.list())
