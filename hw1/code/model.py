import torch
import torch.nn as nn
from torchvision import models


class CustomResnextModel(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        # Docstring

        super(CustomResnextModel, self).__init__()

        if pretrained:
            self.base_model = models.resnext101_64x4d(
                weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
            )
        else:
            self.base_model = models.resnext101_64x4d(weights=None)

        self.base_model.fc = nn.Sequential(
            nn.Linear(
                in_features=self.base_model.fc.in_features,
                out_features=2048
            ),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        return x

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))

    def save_weight(self, path):
        torch.save(self.state_dict(), path)