import os
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import OrderedDict
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNeXt50_32X4D_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN


class CustomedModel(nn.Module):
    def __init__(
        self,
        anchor_generator,
        roi_pooler,
        mask_roi_pooler,
        num_classes: int,
        pretrained=True
    ):
        super(CustomedModel, self).__init__()

        # trainable backbone layers
        backbone_with_fpn = resnet_fpn_backbone(
            backbone_name="resnext50_32x4d",
            weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None,
            trainable_layers=5,
        )

        rpn_head = RPNHead(
            in_channels=256,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            conv_depth=5
        )

        model = MaskRCNN(
            backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        model.roi_heads.positive_fraction = 0.25

        in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 512
        model.roi_heads.mask_predictor = CustomedMaskRCNNPredictor(
            in_features, hidden_layer, num_classes=num_classes
        )

        self.model = model

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

    def load_pretrained_weight(
        self,
        pretrained_weight_path: str,
        weight_only=True,
        device="cuda"
    ):
        if os.path.exists(pretrained_weight_path):
            stat_dict = torch.load(
                pretrained_weight_path,
                weights_only=weight_only,
                map_location=device
            )
            new_state_dict = OrderedDict()
            for k, v in stat_dict.items():
                name = k.replace("model.", "")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            print("Pretrained weight not exist.")

    def save_model(self, save_model_path: str):
        torch.save(self.model.state_dict(), save_model_path)


class CustomedMaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(
                        in_channels, dim_reduced, 2, 2, 0)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv6_mask", nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv7_mask", nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(
                        dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu"
                )


class CustomedMaskRCNNPredictorv2(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        layers = []
        layers.append(("deconv", nn.ConvTranspose2d(
            in_channels, dim_reduced, 2, 2, 0)))
        layers.append(("gn0", nn.GroupNorm(32, dim_reduced)))
        layers.append(("relu0", nn.ReLU(inplace=True)))

        for i in range(3):
            layers.append((f"conv{i}", nn.Conv2d(
                dim_reduced, dim_reduced, 3, padding=1)))
            layers.append((f"gn{i+1}", nn.GroupNorm(32, dim_reduced)))
            layers.append((f"relu{i+1}", nn.ReLU(inplace=True)))
            layers.append((f"drop{i}", nn.Dropout2d(0.1)))

        layers.append(
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1))
        )
        super().__init__(OrderedDict(layers))

        for n, p in self.named_parameters():
            if "weight" in n and p.dim() > 1:
                nn.init.kaiming_normal_(
                    p, mode="fan_out", nonlinearity="relu"
                )
