import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data.fashionIQ import FashionIQDataset, FashionIQTestDataset, FashionIQTestQueryDataset
from data.fashion200k import Fashion200kDataset, Fashion200kTestDataset, Fashion200kTestQueryDataset
from data.shoes import ShoesDataset, ShoesTestDataset, ShoesTestQueryDataset
from data.collate_fns import PaddingCollateFunction, PaddingCollateFunctionTest
from language import AbstractBaseVocabulary

DEFAULT_VOCAB_PATHS = {
    **dict.fromkeys(FashionIQDataset.all_codes(), FashionIQDataset.vocab_path()),
    **dict.fromkeys(ShoesDataset.all_codes(), ShoesDataset.vocab_path()),
    **dict.fromkeys(Fashion200kDataset.all_codes(), Fashion200kDataset.vocab_path())
}


def _random_indices(dataset_length, limit_size):
    return np.random.randint(0, dataset_length, limit_size)


def train_dataset_factory(transforms, config):
    image_transform = transforms['image_transform']
    text_transform = transforms['text_transform']
    #text_transform = None
    id_transform = transforms['id_transform']
    dataset_code = config['dataset']
    use_subset = config.get('use_subset', False)
    if config['text_encoder'] == 'bert':
        text_transform = None

    if FashionIQDataset.code() in dataset_code:
        dataset_clothing_split = dataset_code.split("_")
        if len(dataset_clothing_split) == 1:
            raise ValueError("Please specify clothing type for this dataset: fashionIQ_[dress_type]")
        #clothing_type = dataset_clothing_split[1]
        #dataset = FashionIQDataset(split='train', clothing_type=clothing_type, img_transform=image_transform,
        #                           text_transform=text_transform)
        dataset1 = FashionIQDataset(split='train', clothing_type='dress', img_transform=image_transform,
                                   text_transform=text_transform, id_transform=id_transform)
        dataset2 = FashionIQDataset(split='train', clothing_type='shirt', img_transform=image_transform,
                                   text_transform=text_transform, id_transform=id_transform)
        dataset3 = FashionIQDataset(split='train', clothing_type='toptee', img_transform=image_transform,
                                   text_transform=text_transform, id_transform=id_transform)
        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])
    elif ShoesDataset.code() in dataset_code:
        dataset = ShoesDataset(split='train', clothing_type=None, img_transform=image_transform,
                               text_transform=text_transform, id_transform=id_transform)
    elif Fashion200kDataset.code() in dataset_code:
        dataset = Fashion200kDataset(split='train', clothing_type=None, img_transform=image_transform,
                                     text_transform=text_transform, id_transform=id_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    if use_subset:
        return Subset(dataset, _random_indices(len(dataset), 1000))

    return dataset


def test_dataset_factory(transforms, config, split='val'):
    image_transform = transforms['image_transform']
    text_transform = transforms['text_transform']
    dataset_code = config['dataset']
    use_subset = config.get('use_subset', False)

    if config['text_encoder'] == 'bert':
        text_transform = None

    if FashionIQDataset.code() in dataset_code:
        dataset_clothing_split = dataset_code.split("_")
        if len(dataset_clothing_split) == 1:
            raise ValueError("Please specify clothing type for this dataset: fashionIQ_[dress_type]")
        clothing_type = dataset_clothing_split[1]
        test_samples_dataset = FashionIQTestDataset(split=split, clothing_type=clothing_type,
                                                    img_transform=image_transform, text_transform=text_transform)
        test_query_dataset = FashionIQTestQueryDataset(split=split, clothing_type=clothing_type,
                                                       img_transform=image_transform, text_transform=text_transform)
    elif ShoesDataset.code() in dataset_code:
        test_samples_dataset = ShoesTestDataset(split=split, clothing_type=None,
                                                    img_transform=image_transform, text_transform=text_transform)
        test_query_dataset = ShoesTestQueryDataset(split=split, clothing_type=None,
                                                       img_transform=image_transform, text_transform=text_transform)

    elif Fashion200kDataset.code() in dataset_code:
        test_samples_dataset = Fashion200kTestDataset(split='eval', clothing_type=None,
                                                    img_transform=image_transform, text_transform=text_transform)
        test_query_dataset = Fashion200kTestQueryDataset(split='test', clothing_type=None,
                                                       img_transform=image_transform, text_transform=text_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    if use_subset:
        return {"samples": Subset(test_samples_dataset, _random_indices(len(test_samples_dataset), 1000)),
                "query": Subset(test_query_dataset, _random_indices(len(test_query_dataset), 1000))}

    return {"samples": test_samples_dataset,
            "query": test_query_dataset}


def train_dataloader_factory(dataset, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = config.get('shuffle', True)
    # TODO: remove this
    drop_last = True

    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn, drop_last=drop_last)


def test_dataloader_factory(datasets, config, collate_fn=None):
    batch_size = config['batch_size']
    #batch_size = 32
    num_workers = config.get('num_workers', 16)
    shuffle = False

    return {
        'query': DataLoader(datasets['query'], batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn),
        'samples': DataLoader(datasets['samples'], batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    }


def create_dataloaders(image_transform, text_transform, id_transform, configs):
    train_dataset = train_dataset_factory(
        transforms={'image_transform': image_transform['train'], 'text_transform': text_transform['train'],
                    'id_transform': id_transform['train']},config=configs)
    test_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform['val']},
        config=configs)
    if configs['text_encoder'] == 'bert':
        padding_idx = 1
    elif configs['text_encoder'] == 'lstm':
        padding_idx = AbstractBaseVocabulary.pad_id()
    collate_fn = PaddingCollateFunction(padding_idx=padding_idx)
    collate_fn_test = PaddingCollateFunctionTest(padding_idx=padding_idx)
    train_dataloader = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn)
    train_dataloader2 = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn)
    test_dataloaders = test_dataloader_factory(datasets=test_datasets, config=configs, collate_fn=collate_fn_test)
    train_val_dataloaders = None
    return train_dataloader, test_dataloaders, train_dataloader2
