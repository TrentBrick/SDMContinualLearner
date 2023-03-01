# ConvMixer from Patches are all you need with timm training augmentations: 
# https://github.com/locuslab/convmixer-cifar10/blob/main/train.py


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from types import SimpleNamespace
from models import BaseModel, Residual
import numpy as np
import time

class ConvMixer(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.features = nn.Sequential(
            nn.Conv2d(3, params.hdim, kernel_size=params.patch_size, stride=params.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(params.hdim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(params.hdim, params.hdim, params.kernel_size, groups=params.hdim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(params.hdim)
                    )),
                    nn.Conv2d(params.hdim, params.hdim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(params.hdim)
            ) for i in range(params.nblocks)],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(params.hdim, params.output_size)

    def forward(self, x, log_metrics=True, output_model_data=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x