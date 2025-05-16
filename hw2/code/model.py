import torch.nn as nn

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CustomModel(nn.Module):

    def __init__(self, num_classes, anchor_generator, roi_pooler, pretrained=True):
        super(CustomModel, self).__init__()
        self.base_model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.base_model.roi_heads.nms_thresh = 0.3
        self.base_model.roi_heads.score_thresh = 0.5
        self.base_model.roi_heads.positive_fraction = 0.25

        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.base_model(images, targets)
        else:
            return self.base_model(images)
