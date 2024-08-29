import json
import torch
from torchvision import transforms

class Str2NumId(object):
    def __init__(self, config):

        if config['dataset'] == 'shoes':
            f = open('data/shoes/shoes_labels.json', 'r')
        elif config['dataset'] == 'fashion200k':
            f = open('data/fashion200k/fashion200k_labels.json', 'r')
        else:
            f = open('data/fashionIQ/fashioniq_labels.json', 'r')
        self.js = json.load(f)

    def __call__(self, str_id):

        num_id = self.js[str_id]
        return num_id

def id_transform_factory(config: dict):

    return {
        'train':transforms.Compose([Str2NumId(config)]),
        'val':None
    }

