from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
#from models.resnet import resnet18, resnet50
from transformers import Swinv2ForImageClassification

from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder


class ResNet18Layer4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True):
        super().__init__()
        self._model = resnet18(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 512, 'layer3': 256, 'layer2': 128, 'layer1': 64}


class ResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']
        #self.bn = nn.BatchNorm1d(lower_feature_shape)
        #self.norm_scale = torch.nn.Parameter(torch.ones(1)*4, requires_grad=True)
        #self.fc1 = nn.Linear(self.lower_feature_shape, self.lower_feature_shape)

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
#        layer4_out = self.dropout(layer4_out)
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        #layer4_out = x.clone()
        #x = self.fc1(x)
        #x = self.bn(x)
        x_norm = self.fc(x)
        #x = self.dropout(x_norm)
        x_norm = F.normalize(x_norm) * self.norm_scale

        return x_norm#, layer4_out

class GAPResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(x) * self.norm_scale

        return x


class ResNet50Layer4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True):
        super().__init__()
        self._model = resnet50(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256}


class ResNet50Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x_norm = self.fc(x)
        x_norm = F.normalize(x_norm) * self.norm_scale

        return x_norm, x

class Swinv2TinyLower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.model = self.model.swinv2
        breakpoint()

    def forward(self, x):
        x = self.model.encoder
        return
    def layer_shapes(self):
        return {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256}

