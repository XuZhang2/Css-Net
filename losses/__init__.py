from losses.batch_based_classification_loss import BatchBasedClassificationLoss
from losses.triplet_loss import BatchHard
from losses.one_triplet_loss import One2OneTripletLoss
import torch.nn as nn


def loss_factory(config):
    if config['metric_loss'] == "batch_based_classification_loss":
        metric_loss = BatchBasedClassificationLoss(config['batch_size'])
    elif config['metric_loss'] == 'batch_hard':
        metric_loss = BatchHard(m=0.2)
    one2one_loss = One2OneTripletLoss()
    return {
        'metric_loss': metric_loss,
        'one2one_loss': one2one_loss,
        'classify_loss': nn.CrossEntropyLoss(),
    }
