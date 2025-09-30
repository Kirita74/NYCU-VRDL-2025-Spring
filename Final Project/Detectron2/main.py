"""
This script is designed to train a Detectron2 model for 
    instance segmentation on the Sartorius Cell dataset.
It uses a custom evaluator to compute the mean Average Precision (MaP) 
    based on Intersection over Union (IoU) for the model's predictions.
"""

from detectron2.evaluation.evaluator import DatasetEvaluator
import pycocotools.mask as mask_util
from detectron2.utils.events import get_event_storage
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import numpy as np
import os
import warnings

from dataset import register_datasets
register_datasets()

warnings.filterwarnings("ignore")

class MAPIOUEvaluator(DatasetEvaluator):
    """
    Custom evaluator to compute mean Average Precision (MaP) based on Intersection over Union (IoU).
    This evaluator processes the outputs of the model 
        and compares them with the ground truth annotations.
    """

    def __init__(self, dataset_name):
        """Initialize the evaluator with the dataset name."""

        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {
            item['image_id']: item['annotations'] for item in dataset_dicts
        }

    def reset(self):
        """Reset the evaluator state before processing a new batch of inputs."""

        self.scores = []

    def process(self, inputs, outputs):
        """
        Process the model outputs and compute scores based on the ground truth annotations.
        Args:
            inputs (list[dict]): List of input data dictionaries.
            outputs (list[dict]): List of output data dictionaries from the model.
        """

        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        """
        Evaluate the model's performance by calculating the mean Average Precision (MaP) IoU.
        Returns:
            dict: A dictionary containing the mean Average Precision (MaP) IoU score.
        """

        result = {"MaP IoU": np.mean(self.scores)}
        try:
            current_iter = get_event_storage().iter
            epoch = int(current_iter // (7760 / 8))
        except BaseException:
            current_iter = -1
            epoch = -1

        with open("map.txt", "a") as f:
            f.write(
                f"[Epoch {epoch} | Iter {current_iter}] MaP IoU = {result['MaP IoU']:.6f}\n")

        return result


def convert_instances_to_annotations(masks, labels):
    """
    Convert model output masks and labels into COCO-style annotations.
    Args:
        masks (torch.Tensor): A tensor of shape (N, H, W) containing binary masks.
        labels (torch.Tensor): A tensor of shape (N,) containing class labels.
    Returns:
        list[dict]: A list of dictionaries, 
            each containing the segmentation, category_id, iscrowd, bbox, and bbox_mode.
    """

    annotations = []
    for mask, label in zip(masks.cpu().numpy(), labels.cpu().numpy()):
        ys, xs = np.where(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        bbox = [x0, y0, x1, y1]
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotations.append({
            "segmentation": rle,
            "category_id": int(label),
            "iscrowd": 0,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS
        })
    return annotations


def precision_at(threshold, iou):
    """
    Calculate true positives (TP), false positives (FP), 
        and false negatives (FN) based on IoU threshold.
    Args:
        threshold (float): The IoU threshold to determine matches.
        iou (np.ndarray): A 2D array of shape (N, M) 
            where N is the number of predicted masks and M is the number of ground truth masks.
    Returns:
        tuple: A tuple containing the number of true positives, 
            false positives, and false negatives.
    """

    matches = iou > threshold
    tp = np.sum(matches, axis=1) == 1
    fp = np.sum(matches, axis=0) == 0
    fn = np.sum(matches, axis=1) == 0
    return np.sum(tp), np.sum(fp), np.sum(fn)


def score(pred, targ):
    """
    Calculate the mean Average Precision (MaP) IoU score for the model's predictions.
    Args:
        pred (dict): The model's predictions containing 'instances' with predicted masks.
        targ (list[dict]): The ground truth annotations for the image.
    Returns:
        float: The mean Average Precision (MaP) IoU score.
    """

    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F'))
                 for p in pred_masks]
    enc_targs = list(map(lambda x: x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        prec.append(p)
    return np.mean(prec)


class Trainer(DefaultTrainer):
    """
    Custom trainer class that extends DefaultTrainer to use the custom MAPIOUEvaluator.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Build the evaluator for the given dataset.
        Args:
            cfg (CfgNode): The configuration object.
            dataset_name (str): The name of the dataset to evaluate.
            output_folder (str, optional): The folder to save evaluation results.
        Returns:
            MAPIOUEvaluator: An instance of the custom MAPIOUEvaluator.
        """

        return MAPIOUEvaluator(dataset_name)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sartorius_Cell_train",)
cfg.DATASETS.TEST = ("sartorius_Cell_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "cross5fold/fold1/model_0016999.pth"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (7000, 9000)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000
cfg.TEST.AUG.ENABLED = True
# cfg.TEST.AUG.MIN_SIZES = (440, 480, 520, 560, 580, 620) # for patch dataset
cfg.TEST.AUG.MIN_SIZES = (640, 672, 704, 736, 768, 800)  # for full dataset
cfg.TEST.AUG.MAX_SIZE = 1333
cfg.TEST.AUG.FLIP = True
# cfg.INPUT.MIN_SIZE_TRAIN = (440, 480, 520, 560, 580, 620) # for patch dataset
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)   # for full dataset
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.sample_style = "choice"
cfg.OUTPUT_DIR = "output_fold1_patched"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
