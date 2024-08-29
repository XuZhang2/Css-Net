from evaluators.abc import AbstractBaseEvaluator
import torch.nn.functional as F
import torch
import torch.nn as nn


class SimpleEvaluator_cosmo(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, top_k=(1, 10, 50), visualizer=None, configs=None):
        super().__init__(models, dataloaders, top_k, visualizer)
        self.lower_image_encoder = self.models['lower_image_encoder']
        self.upper_image_encoder = self.models['upper_image_encoder']
        self.upper_image_encoder3 = self.models['upper_image_encoder3']
        self.upper_image_encoder_t = self.models['upper_image_encoder_t']
        self.upper_image_encoder_t3 = self.models['upper_image_encoder_t3']
        self.text_encoder = self.models['text_encoder']
        self.text_transformation = self.models['text_transformation']
        self.text_transformation3 = self.models['text_transformation3']
        self.compositor = self.models['layer4']
        self.compositor3 = self.models['layer3']
        self.configs = configs

    def _extract_image_features(self, images):
        mid_features, shallows = self.lower_image_encoder(images)

        original_features = self.upper_image_encoder(mid_features)
        #original_features = F.normalize(original_features)

        original_features2 = self.upper_image_encoder_t(mid_features)
        original_features4 = self.upper_image_encoder_t3(shallows[0])
        original_features3 = self.upper_image_encoder3(shallows[0])
        #original_features2 = self.targ_fc(mid_features)
        #original_features = self.all_fc(torch.cat((original_features, original_features2), dim=1))
        return original_features, original_features2, original_features3, original_features4

    def _extract_original_and_composed_features(self, images, modifiers, len_modifiers, attn_mask):

        mid_image_features, shallows = self.lower_image_encoder(images)
        image_features = self.upper_image_encoder_t(mid_image_features)
        image_features3 = self.upper_image_encoder_t3(shallows[0])
        if self.configs['text_encoder'] == 'bert':
            text_feat = self.text_encoder(modifiers, attn_mask)
            new_text = self.text_transformation(text_feat, image_features)
            new_text3 = self.text_transformation3(text_feat, image_features3)
            composed_features2 = new_text
            composed_features4 = new_text3
            text_features = self.text_transformation(text_feat)
            text_features3 = self.text_transformation3(text_feat)

        composed_features, _= self.compositor(mid_image_features, text_features)
        composed_features = self.upper_image_encoder(composed_features)
        composed_features3, _= self.compositor3(shallows[0], text_features3)
        composed_features3 = self.upper_image_encoder3(composed_features3)


        return composed_features, composed_features2, composed_features3, composed_features4

class SimpleEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, top_k=(1, 10, 50), visualizer=None, configs=None):
        super().__init__(models, dataloaders, top_k, visualizer)
        self.lower_image_encoder = self.models['lower_image_encoder']
        self.upper_image_encoder = self.models['upper_image_encoder']
        # self.upper_image_encoder3 = self.models['upper_image_encoder3']
        # self.upper_image_encoder_t = self.models['upper_image_encoder_t']
        # self.upper_image_encoder_t3 = self.models['upper_image_encoder_t3']
        self.text_encoder = self.models['text_encoder']
        self.text_transformation = self.models['text_transformation']
        # self.text_transformation3 = self.models['text_transformation3']
        self.compositor = self.models['layer4']
        # self.compositor3 = self.models['layer3']
        self.configs = configs

    def _extract_image_features(self, images):
        mid_features, _ = self.lower_image_encoder(images)
        original_features = self.upper_image_encoder(mid_features)
        #original_features = self.stage2_encoder(original_features)
        return F.normalize(original_features)

    def _extract_original_and_composed_features(self, images, modifiers, len_modifiers, attn_mask):
        mid_image_features, shallows = self.lower_image_encoder(images)
        #image_features = self.upper_image_encoder_t(mid_image_features)
        # image_features3 = self.upper_image_encoder_t3(shallows[0])
        if self.configs['text_encoder'] == 'bert':
            text_feat = self.text_encoder(modifiers, attn_mask)
            # new_text = self.text_transformation(text_feat, image_features)
            # new_text3 = self.text_transformation3(text_feat, image_features3)
            # composed_features2 = new_text
            # composed_features4 = new_text3
            text_features = self.text_transformation(text_feat)
            # text_features3 = self.text_transformation3(text_feat)

        composed_features, _= self.compositor(mid_image_features, text_features)
        composed_features = self.upper_image_encoder(composed_features)
        # composed_features3, _= self.compositor3(shallows[0], text_features3)
        # composed_features3 = self.upper_image_encoder3(composed_features3)
        return composed_features


