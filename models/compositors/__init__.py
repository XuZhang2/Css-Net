from models.compositors.global_style_models import GlobalStyleTransformer2
from models.compositors.transformers import DisentangledTransformer
import torch.nn as nn


def global_styler_factory(code, feature_size, text_feature_size):
    if code == GlobalStyleTransformer2.code():
        return GlobalStyleTransformer2(feature_size, text_feature_size)
    else:
        raise ValueError("{} not exists".format(code))


def transformer_factory(feature_sizes, configs):
    text_feature_size = feature_sizes['text_feature_size']
    num_heads = configs['num_heads']
    feature_size = feature_sizes['layer4']
    global_styler_code = configs['global_styler']
    global_styler = global_styler_factory(global_styler_code, feature_sizes['layer4'], text_feature_size)
    return DisentangledTransformer(feature_sizes['layer4'], text_feature_size, num_heads=num_heads,
                                              norm='in', global_styler=global_styler)
    #layer4 = {'layer4': DisentangledTransformer(feature_sizes['layer4'], text_feature_size, num_heads=num_heads,
    #                                          norm='in', global_styler=None)}
    #layer4 = {'layer4': nn.Sequential(*[DisentangledTransformer(feature_sizes['layer4'], text_feature_size, num_heads=num_heads,
    #                                          norm='in', global_styler=None),
    #                     DisentangledTransformer(feature_sizes['layer4'], text_feature_size, num_heads=num_heads,
    #                                          norm='bn', global_styler=None),
    #                     nn.Conv2d(feature_size*2, feature_size, 1),
    #                     nn.BatchNorm2d(feature_size)])}
    return layer4
