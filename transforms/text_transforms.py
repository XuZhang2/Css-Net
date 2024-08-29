from typing import List

import torch
from torchvision import transforms

from language import AbstractBaseVocabulary
from transformers import BertTokenizer
from transformers import AutoTokenizer

from transforms.text_noise import random_deletion, random_insertion, random_swap

class ToIds(object):
    def __init__(self, vocabulary: AbstractBaseVocabulary):
        self.vocabulary = vocabulary

    def __call__(self, text: str) -> List[int]:
        return self.vocabulary.convert_text_to_ids(text)


class ToLongTensor(object):
    def __call__(self, ids: List[int]) -> torch.LongTensor:
        return torch.LongTensor(ids)

class BertTokenizerHandle(object):
    def __init__(self):
        model_name = 'bert-base-uncased'
        #self.tokenizer = BertTokenizer.from_pretrained(model_name,local_files_only=True)
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        #print('none')

    def __call__(self, text: str) -> List[int]:
        x = self.tokenizer.encode(text, return_tensors='pt').squeeze(0)
#       tokens = bert_tokenizer.batch_encode_plus(nl, padding='longest',
#                                                  return_tensors='pt')
        return x


def text_transform_factory(config: dict):
    vocabulary = config['vocabulary']
    model = config['model']

    if model == 'bert':
        return {
        'train': transforms.Compose([BertTokenizerHandle(), ToLongTensor()]),
        'val': transforms.Compose([BertTokenizerHandle(), ToLongTensor()])
        }
    return {
        'train': transforms.Compose([ToIds(vocabulary), ToLongTensor()]),
        'val': transforms.Compose([ToIds(vocabulary), ToLongTensor()])
    }
