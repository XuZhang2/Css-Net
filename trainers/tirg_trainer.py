from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet
import numpy as np
from mem_bank import Memory
from torch.cuda.amp import autocast as autocast, GradScaler

class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.length = len(loader)
        self.iter = None

    def __len__(self):
        return self.length

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

class TIRGTrainer(AbstractBaseTrainer):
    def __init__(self, models, train_dataloader,train_dataloader2, criterions, optimizers, lr_schedulers, config,
                 train_loggers, val_loggers, evaluator, warmup_schedulers, *args, **kwargs):
        super().__init__(models, train_dataloader, train_dataloader2, criterions, optimizers, lr_schedulers, config,
                         train_loggers, val_loggers, evaluator, warmup_schedulers, *args, **kwargs)
        self.lower_image_encoder = self.models['lower_image_encoder']
        self.upper_image_encoder = self.models['upper_image_encoder']
        self.upper_image_encoder3 = self.models['upper_image_encoder3']
        self.upper_image_encoder_t = self.models['upper_image_encoder_t']
        self.upper_image_encoder_t3 = self.models['upper_image_encoder_t3']
        self.text_encoder = self.models['text_encoder']
        self.text_transformation = self.models['text_transformation']
        self.text_transformation3 = self.models['text_transformation3']
        self.compositor = self.models['layer4']
        #self.compositor2 = self.models['layer2']
        self.compositor3 = self.models['layer3']

        self.metric_loss = self.criterions['metric_loss']
        self.classify_loss = self.criterions['classify_loss']
        self.one2one_loss = self.criterions['one2one_loss']
        self.config = config
        #self.memory = Memory(config['feature_size'], config['text_feature_size'], config['num_instance'], self.metric_loss)
        self.scaler = GradScaler()

    def train_one_epoch(self, epoch):

        stage = 1 if epoch < self.config['stage1_epoch'] else 2

        average_meter_set = AverageMeterSet()
        if stage == 1:
            self.adjust_learning_rate(self.optimizers, epoch, stage)
        else:
            self.adjust_learning_rate(self.optimizers, epoch-self.config['stage1_epoch'], stage)

        '''for record'''
        key = 'lower_image_encoder'
        lr = self.optimizers[key].param_groups[0]['lr']
        average_meter_set.update('image_lr', lr)
        key = 'text_encoder'
        text_lr = self.optimizers[key].param_groups[0]['lr']
        average_meter_set.update('text lr', text_lr)
        key = 'layer4'
        compositor_lr = self.optimizers[key].param_groups[0]['lr']
        average_meter_set.update('compositor_lr', compositor_lr)
        '''record over'''
        #text_dataloader = IterLoader(self.train_dataloader2)

        #self.train_dataloader.dataset.generate_random_train_queries_(n_modifications_per_image=3)
        train_dataloader = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        for batch_idx, (ref_images, tar_images, modifiers, len_modifiers, ref_ids, tar_ids, attn_mask) in enumerate(train_dataloader):
        #len_loader = tqdm(range(len(text_dataloader)//2), desc="Epoch {}".format(epoch))
        #for i in len_loader:
            ref_images, tar_images = ref_images.to(self.device), tar_images.to(self.device)
            images = torch.cat((ref_images, tar_images), dim=0)
            len_modifiers = len_modifiers.to(self.device)
            attn_mask = attn_mask.to(self.device)
            ref_ids, tar_ids = ref_ids.to(self.device), tar_ids.to(self.device)
            ids = torch.cat((ref_ids, tar_ids), dim=0)
            '''

            ref_images, tar_images, modifiers, \
                len_modifiers, ref_ids, tar_ids, attn_mask = text_dataloader.next()
            ref_images2, tar_images2, modifiers2, \
                len_modifiers2, ref_ids2, tar_ids2, attn_mask2 = text_dataloader.next()
            '''

            self._reset_grad()
            '''This is for stage 1'''
            if stage == 1:
                features, _ = self.lower_image_encoder(images)
                norm_feat = self.upper_image_encoder(features)
                probs = self.image_classifier(norm_feat, ids)

                loss_l = self.classify_loss(probs, ids)
                loss_l.backward()
                average_meter_set.update('classify_loss', loss_l.item())

            elif self.config['model'] == "cosmo":
                #with autocast():
                if True:
                    tar_mid_features, tar_shallows = self.lower_image_encoder(tar_images)
                    #tar_mid_features2, _ = self.lower_image_encoder2(tar_images)
                    #tar_mid_features2 = tar_mid_features#.clone().detach()

                    norm_tar_protos = self.upper_image_encoder(tar_mid_features)
                    norm_tar_protos3 = self.upper_image_encoder3(tar_shallows[0])
                    norm_tar_protos_t = self.upper_image_encoder_t(tar_mid_features)
                    norm_tar_protos_t3 = self.upper_image_encoder_t3(tar_shallows[0])
                    #norm_tar_protos2 = self.upper_image_encoder2(tar_shallows[1])

                    # Encode and Fuse Reference Images with Texts
                    ref_mid_features, ref_shallows = self.lower_image_encoder(ref_images)

                    norm_ref_protos = self.upper_image_encoder_t(ref_mid_features)
                    norm_ref_protos3 = self.upper_image_encoder_t3(ref_shallows[0])

                    if self.config['text_encoder'] == 'bert':
                        text_feat = self.text_encoder(modifiers, attn_mask)

                        new_text = self.text_transformation(text_feat, norm_ref_protos)
                        new_text3 = self.text_transformation3(text_feat, norm_ref_protos3)
                        text_features = self.text_transformation(text_feat)
                        #text_features2 = self.text_transformation2(text_feat)
                        text_features3 = self.text_transformation3(text_feat)

                    composed_ref_protos, _= self.compositor(ref_mid_features, text_features)
                    #composed_ref_protos2, _= self.compositor2(ref_shallows[1], text_features2)
                    composed_ref_protos3, _= self.compositor3(ref_shallows[0], text_features3)
                    norm_com_protos = self.upper_image_encoder(composed_ref_protos)
                    #norm_com_protos2 = self.upper_image_encoder2(composed_ref_protos2)
                    norm_com_protos3 = self.upper_image_encoder3(composed_ref_protos3)

                   # Compute Loss
                    if self.config['metric_loss'] == 'batch_hard':
                        loss = self.metric_loss(composed_ref_protos, tar_protos, ref_ids, tar_ids)
                    else:
                        loss1, loss3, losst, losst3, losskl = self.metric_loss(norm_com_protos, norm_tar_protos,
                            norm_com_protos3, norm_tar_protos3, new_text, norm_tar_protos_t, new_text3, norm_tar_protos_t3)
                    '''optional add'''
                    average_meter_set.update('layer4_loss', loss1.item())
                    average_meter_set.update('layer3_loss', loss3.item())
                    average_meter_set.update('layert_loss', losst.item())
                    average_meter_set.update('layert3_loss', losst3.item())
                    average_meter_set.update('layerkl_loss', losskl.item())
                    #loss = loss1
                    loss = loss1+loss3+losst+losst3+losskl
                    #loss = loss1#+loss3+losskl

                loss.backward()
                #self.scaler.scale(loss).backward()

            self._update_grad()
            #self.scaler.update()
        train_results = average_meter_set.averages()
        torch.cuda.empty_cache()
        train_dataloader.close()
        return train_results

    @classmethod
    def code(cls) -> str:
        return 'tirg'
