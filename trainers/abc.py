import abc
from abc import ABC
from typing import Sequence, Tuple, Any

import torch
from torch import nn as nn


class AbstractBaseLogger(ABC):
    @abc.abstractmethod
    def log(self, log_data: dict, step: int, commit: bool) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, log_data: dict, step: int) -> None:
        raise NotImplementedError


class LoggingService(object):
    def __init__(self, loggers: Sequence[AbstractBaseLogger]):
        self.loggers = loggers

    def log(self, log_data: dict, step: int, commit=False):
        for logger in self.loggers:
            logger.log(log_data, step, commit=commit)

    def complete(self, log_data: dict, step: int):
        for logger in self.loggers:
            logger.complete(log_data, step)


class AbstractBaseMetricLoss(nn.Module, ABC):
    @abc.abstractmethod
    def forward(self, ref_features: torch.Tensor, tar_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseImageLowerEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def layer_shapes(self):
        raise NotImplementedError


class AbstractBaseImageUpperEncoder(nn.Module, abc.ABC):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__()
        self.lower_feature_shape = lower_feature_shape
        self.feature_size = feature_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AbstractBaseTextEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__()
        self.feature_size = feature_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseCompositor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, mid_image_features, text_features, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractGlobalStyleTransformer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, normed_x, t, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseTrainer(ABC):
    def __init__(self, models, train_dataloader, train_dataloader2, criterions, optimizers, lr_schedulers, config,
                 train_loggers, val_loggers, evaluator, warmup_schedulers, *args, **kwargs):
        self.models = models
        self.models2 = models.copy()
        self.models3 = models.copy()
        self.models4 = models.copy()
        self.train_dataloader = train_dataloader
        self.train_dataloader2 = train_dataloader2
        self.criterions = criterions
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.config = config
        self.num_epochs = config['epoch']
        self.train_logging_service = LoggingService(train_loggers)
        self.val_logging_service = LoggingService(val_loggers)
        self.evaluator = evaluator
        self.warmup_schedulers = warmup_schedulers
        #self.train_evaluator = train_evaluator
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_epoch = kwargs['start_epoch'] if 'start_epoch' in kwargs else 0
        self.base_lr = 32e-4
        self.finetune_lr = 2e-5

    def train_one_epoch(self, epoch) -> dict:
        raise NotImplementedError

    def run(self) -> dict:
        self._load_models_to_device()
        for epoch in range(self.start_epoch, self.num_epochs):
            for phase in ['train', 'val']:
            #for phase in ['val']:
                if phase == 'train':
                    self._to_train_mode()
                    train_results = self.train_one_epoch(epoch)
                    self.train_logging_service.log(train_results, step=epoch)
                    print(train_results)
                else:
                    self._to_eval_mode()
                    if type(self.evaluator) == list:
                        all_val_results = {}
                        for i in range(len(self.evaluator)):
                            e = self.evaluator[i]
                            val_results, _ = e.evaluate(epoch)
                            for item in val_results.items():
                                if i == 0:
                                    all_val_results['shirt_'+item[0]]=item[1]
                                elif i == 1:
                                    all_val_results['dress_'+item[0]]=item[1]
                                elif i == 2:
                                    all_val_results['toptee_'+item[0]]=item[1]
                        model_state_dicts = self._get_state_dicts(self.models)
                        optimizer_state_dicts = self._get_state_dicts(self.optimizers)
                        all_val_results['model_state_dict'] = model_state_dicts
                        all_val_results['optimizer_state_dict'] = optimizer_state_dicts
                        self.val_logging_service.log(all_val_results, step=epoch, commit=True)
                    else:
                        val_results, _ = self.evaluator.evaluate(epoch)
                        #train_val_results = self.train_evaluator.evaluate(epoch)
                        #self.train_logging_service.log(train_val_results, step=epoch)
                        model_state_dicts = self._get_state_dicts(self.models)
                        optimizer_state_dicts = self._get_state_dicts(self.optimizers)
                        val_results['model_state_dict'] = model_state_dicts
                        val_results['optimizer_state_dict'] = optimizer_state_dicts
                        self.val_logging_service.log(val_results, step=epoch, commit=True)

        return self.models

    def _load_models_to_device(self):
        for model in self.models.values():
            model.to(self.device)

    def _to_train_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].train()

    def _to_eval_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].eval()

    def _reset_grad(self, keys=None):
        keys = keys if keys else self.optimizers.keys()
        for key in keys:
            self.optimizers[key].zero_grad()

    def _update_grad(self, keys=None, exclude_keys=None):
        keys = keys if keys else list(self.optimizers.keys())
        if exclude_keys:
            keys = [key for key in keys if key not in exclude_keys]
        for key in keys:
            self.optimizers[key].step()
            #self.scaler.step(self.optimizers[key])

    def _step_schedulers(self, epoch):
        if epoch < self.config['warmup_period']:
            for scheduler in self.lr_schedulers.values():
                scheduler.step(0)
        else:
            for scheduler in self.lr_schedulers.values():
                scheduler.step()
        #for warmup_scheduler in self.warmup_schedulers.values():
        #    warmup_scheduler.dampen()

    def adjust_learning_rate(self, optimizers, epoch, stage):
        if stage == 1:
            for key, optimizer in optimizers.items():
                if key == 'lower_image_encoder' or key == 'upper_image_encoder':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.base_lr
                        param_group['lr'] = self.finetune_lr
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.finetune_lr*10
        else:
            warmup_period = 10
            fast_warmup_period = 5
            decay_step = 20
            decay_step2 = 25
            decay_rate = 0.1
            #if epoch == decay_step or epoch == decay_step2:
            if (epoch-fast_warmup_period)%decay_step == 0 and epoch != fast_warmup_period:
                self.base_lr *= decay_rate
                self.finetune_lr *= decay_rate
            #self.warmup_lr = min(self.finetune_lr, self.finetune_lr*(epoch+1)/warmup_period)
            self.warmup_lr = min(self.finetune_lr, self.finetune_lr*(epoch+1)/fast_warmup_period)
            #self.warmup_lr = min(self.base_lr, self.base_lr*(epoch+1)/fast_warmup_period)
            for key, optimizer in optimizers.items():
                if key == 'layer4':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr
                elif key == 'text_encoder' or key == 'text_encoder2':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr/10
                        param_group['lr'] = self.warmup_lr/10
                elif key == 'text_transformation':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr
                elif key == 'lower_image_encoder' or key=='lower_image_encoder2':
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.warmup_lr
                elif key == 'upper_image_encoder' or key== 'upper_image_encoder2':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr
                elif key == 'projector' or key == 'stage2_encoder':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr
                elif key == 'text_transformation2':
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr
                else:
                    for param_group in optimizer.param_groups:
                        #param_group['lr'] = self.finetune_lr
                        param_group['lr'] = self.warmup_lr

    @staticmethod
    def _get_state_dicts(dict_of_models):
        state_dicts = {}
        for model_name, model in dict_of_models.items():
            if isinstance(model, nn.DataParallel):
                state_dicts[model_name] = model.module.state_dict()
            else:
                state_dicts[model_name] = model.state_dict()
        return state_dicts

    def _load_state_dicts(models, pth):
        pth = os.path.join(pth, 'best.pth')
        state_dicts = torch.load(pth, map_location='cpu')['model_state_dict']
        for key, item in state_dicts.items():
            if key == 'image_classifier':
                continue
            model = models[key]
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(item)
            else:
                model.load_state_dict(item)
            model.eval()
        return models


    @classmethod
    def code(cls) -> str:
        raise NotImplementedError


METRIC_LOGGING_KEYS = {
    'train_loss': 'train/loss',
    'val_loss': 'val/loss',
    'val_correct': 'val/correct'
}
STATE_DICT_KEY = 'state_dict'
