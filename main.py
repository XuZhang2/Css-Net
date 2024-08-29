from data import DEFAULT_VOCAB_PATHS, create_dataloaders
from evaluators import get_evaluator_cls
from evaluators.visualizers import RecallVisualizer
from language import vocabulary_factory
from loggers.file_loggers import BestModelTracker
from loggers.wandb_loggers import WandbSimplePrinter, WandbSummaryPrinter
from losses import loss_factory
from models import create_models
from optimizers import create_optimizers, create_lr_schedulers ,create_warmup_schedulrers
from options import get_experiment_config
from set_up import setup_experiment
from trainers import get_trainer_cls
from transforms import image_transform_factory, text_transform_factory, id_transform_factory

import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)

    vocabulary = vocabulary_factory(config={
        'vocab_path': configs['vocab_path'] if configs['vocab_path'] else DEFAULT_VOCAB_PATHS[configs['dataset']],
        'vocab_threshold': configs['vocab_threshold']
    })
    image_transform = image_transform_factory(config=configs)
    text_transform = text_transform_factory(config={'vocabulary': vocabulary, 'model':configs['text_encoder']})
    id_transform = id_transform_factory(config={'dataset':configs['dataset']})
    train_dataloader, test_dataloaders, train_dataloader2 = create_dataloaders(image_transform, text_transform,
                                                                                   id_transform, configs)
    if configs['dataset'] == 'fashionIQ_shirt':
        configs['dataset'] = 'fashionIQ_dress'
        train_dataloader, test_dataloaders2, train_val_dataloaders = create_dataloaders(image_transform, text_transform,
                                                                                   id_transform, configs)
        configs['dataset'] = 'fashionIQ_toptee'
        train_dataloader, test_dataloaders3, train_val_dataloaders = create_dataloaders(image_transform, text_transform,
                                                                                   id_transform, configs)

    criterions = loss_factory(configs)
    models = create_models(configs, vocabulary)
    optimizers = create_optimizers(models=models, config=configs)
    lr_schedulers = create_lr_schedulers(optimizers, config=configs)
    warmup_schedulers = create_warmup_schedulrers(optimizers, config=configs)
    train_loggers = [WandbSimplePrinter('train/')]
    summary_keys = get_summary_keys(configs)
    best_metric_key = [key for key in summary_keys if '10' in key][0]
    val_loggers = [WandbSimplePrinter('val/'), WandbSummaryPrinter('best_', summary_keys),
                   BestModelTracker(export_root, metric_key=best_metric_key)]
    visualizer = RecallVisualizer(test_dataloaders)
    evaluator = get_evaluator_cls(configs)(models, test_dataloaders, top_k=configs['topk'], visualizer=None, configs=configs)
    if configs['dataset'] == 'fashionIQ_toptee':
        evaluator2 = get_evaluator_cls(configs)(models, test_dataloaders2, top_k=configs['topk'], visualizer=None, configs=configs)
        evaluator3 = get_evaluator_cls(configs)(models, test_dataloaders3, top_k=configs['topk'], visualizer=None, configs=configs)
        evaluator = [evaluator, evaluator2, evaluator3]
    #train_evaluator = get_evaluator_cls(configs)(models, train_val_dataloaders, top_k=configs['topk'],
    #                                             visualizer=None)
    trainer = get_trainer_cls(configs)(models, train_dataloader, train_dataloader2, criterions, optimizers, lr_schedulers,
                                       configs, train_loggers, val_loggers, evaluator, warmup_schedulers,
                                       start_epoch=0)
                                        #train_evaluator,

    trainer.run()
    print(configs)


def get_summary_keys(configs):
    summary_keys = ['recall_@{}'.format(k) for k in configs['topk'].split(",")]
    return summary_keys


if __name__ == '__main__':
    main()
