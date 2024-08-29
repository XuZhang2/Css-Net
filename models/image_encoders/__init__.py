from typing import Tuple
import json
import torch
import torch.nn as nn
from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder
from models.image_encoders.resnet import ResNet18Layer4Lower, ResNet18Layer4Upper, ResNet50Layer4Lower, \
    ResNet50Layer4Upper, Swinv2TinyLower
from models.image_encoders.init import weights_init_classifier, weights_init_classifier_less
import random
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self, config, feature_size):
        super(Classifier, self).__init__()
        if config['dataset'] == 'shoes':
            f = open('data/shoes/shoes_labels.json', 'r')
        elif config['dataset'] == 'fashion200k':
            f = open('data/fashion200k/fashion200k_labels.json', 'r')
        else:
            f = open('data/fashionIQ/fashioniq_labels.json', 'r')
        js = json.load(f)
        self.num_instance = len(js)
        config['num_instance'] = self.num_instance
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(feature_size, feature_size)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        self.bn = nn.BatchNorm1d(feature_size)
        #self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(feature_size, self.num_instance, bias=False)
        self.fc2.apply(weights_init_classifier)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, ids):
        #classify_mask = random.sample(range(self.num_instance), 8)
        #ids = ids.cpu().numpy().tolist()
        #classify_mask = classify_mask+ids
        #classify_mask = ids
        #classify_mask = list(set(classify_mask))
        #mask = torch.zeros(self.num_instance).to(self.device)
        #mask[classify_mask] = 1
            #x = self.avgpool(x)
            #x = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.bn(x)
        #x = self.dropout(x)
        '''not directly'''
        #self.fc2.weight.data = F.normalize(self.fc2.weight.data)*4
        x = self.fc2(x)
        '''
        for i in range(x.shape[0]):
            mask[:] = 0
            classify_mask = random.sample(range(self.num_instance), 8)
            classify_mask = classify_mask+ids
            classify_mask = list(set(classify_mask))
            mask[classify_mask] = 1
            x[i] = x[i].masked_fill(mask==0, -1e9)
        '''
        #x = x.masked_fill(mask==0, -1e9)

        return x


def create_classifier(config, feature_size):

    classifier = Classifier(config, feature_size)
#    classifier.apply(weights_init_classifier)
    return classifier

def create_encoder(feature_size):

    return nn.Sequential(
        nn.Linear(feature_size, feature_size),
        nn.BatchNorm1d(feature_size)
    )

def image_encoder_factory(config: dict, pos: str) -> Tuple[AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder]:
    model_code = config['image_encoder']
    feature_size = config['feature_size']
    pretrained = config.get('pretrained', True)
    norm_scale = config.get('norm_scale', 4)

    if model_code == 'resnet18_layer4':
        lower_encoder = ResNet18Layer4Lower(pretrained)
        lower_feature_shape = lower_encoder.layer_shapes()[pos]
        upper_encoder = ResNet18Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'resnet50_layer4':
        lower_encoder = ResNet50Layer4Lower(pretrained)
        lower_feature_shape = lower_encoder.layer_shapes()[pos]
        upper_encoder = ResNet18Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'swinv2':
        lower_encoder = Swinv2TinyLower(pretrained)
    else:
        raise ValueError("There's no image encoder matched with {}".format(model_code))
    #upper_encoder.apply(weights_init_classifier_less)

    image_classifier = create_classifier(config, feature_size)#lower_feature_shape)
    stage2_encoder = create_encoder(feature_size)
    return lower_encoder, upper_encoder, image_classifier, stage2_encoder
