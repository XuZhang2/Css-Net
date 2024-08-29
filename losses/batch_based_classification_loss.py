import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.abc import AbstractBaseMetricLoss
from dkd import DKD


class BatchBasedClassificationLoss(AbstractBaseMetricLoss):
    def __init__(self, batch_size):
        super().__init__()
        lst = torch.tensor([i for i in range(batch_size)])
        self.sort_matrix = lst.unsqueeze(1).repeat(1, batch_size)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, ref_features, tar_features, ref_features3, tar_features3,
                ref_featurest, tar_featurest, ref_featurest3, tar_featurest3):
        batch_size = ref_features.size(0)
        self.batch_size = batch_size
        device = ref_features.device
        self.device = device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        labels = torch.arange(0, batch_size).long().to(device)
        #labels = self.label_smooth(labels)
        loss1 = F.cross_entropy(pred, labels)
        #pred2 = ref_features2.mm(tar_features2.transpose(0, 1))
        #loss2 = F.cross_entropy(pred2, labels)
        pred3 = ref_features3.mm(tar_features3.transpose(0, 1))
        loss3 = F.cross_entropy(pred3, labels)

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        #kl_loss = DKD(pred, pred3)
        predt = ref_featurest.mm(tar_featurest.transpose(0,1))
        losst = F.cross_entropy(predt, labels)
        predt3 = ref_featurest3.mm(tar_featurest3.transpose(0,1))
        losst3 = F.cross_entropy(predt3, labels)

        mean_pred = F.log_softmax(0.1*pred+10*pred3, dim=1)#+0.5*pred3, dim=1)
        log1 = F.log_softmax(pred, dim=1)
        log3 = F.log_softmax(pred3, dim=1)

        mean_predt = F.log_softmax(1*predt+1*predt3, dim=1)#+0.5*pred3, dim=1)
        logt1 = F.log_softmax(predt, dim=1)
        logt3 = F.log_softmax(predt3, dim=1)

        loss_kl = kl_loss(log1, mean_pred)+kl_loss(log3, mean_pred)# + kl_loss(logt1, mean_predt)+kl_loss(logt3, mean_predt)
        #loss_kl = kl_loss(labels)

        return loss1, loss3, losst, losst3, loss_kl

    @classmethod
    def code(cls):
        return 'batch_based_classification_loss'

    def label_smooth(self, target):

        self.alpha = 0.01
        label_one_hot = F.one_hot(target, self.batch_size).float().to(self.device)
        labels = label_one_hot * (1. - self.alpha) + self.alpha / float(self.batch_size)

        return labels
