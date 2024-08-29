from models.compositors import transformer_factory
from models.image_encoders import image_encoder_factory
from models.text_encoders import text_encoder_factory
from utils.mixins import GradientControlDataParallel
from utils.mixins import BalancedDataParallel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class text_transform(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.img_fc = nn.Linear(feature_size, hidden_size)
        self.text_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.text_bn = nn.BatchNorm1d(hidden_size)
        self.text_fc = nn.Linear(hidden_size, 512)
        self.text_bn2 = nn.BatchNorm1d(hidden_size)
        self.text_fc2 = nn.Linear(hidden_size, 512)
        self.all_conv = nn.Conv1d(hidden_size+feature_size, hidden_size,kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, text, ref_img=None):
        if ref_img is None:
            text = torch.mean(text, dim=1)
            text = self.text_bn2(text)
            text = self.text_fc2(text)
            return text

        img = self.img_fc(ref_img).unsqueeze(-1) #nx768x1
        text_reshaped = text.permute(0,2,1)#nx768x20
        text_reshaped = self.text_conv(text_reshaped).permute(0,2,1)#nx20x768

        attn = torch.matmul(text_reshaped,img)#/np.sqrt(self.hidden_size) #nx20x1
        attn = torch.sigmoid(attn) #nx20x1

        text_reshaped = text.permute(0,2,1)#nx768x20
        #img_reshaped = reshape_text_features_to_concat(ref_img, text_reshaped.shape)#nx512x1x20
        img_reshaped = ref_img.view(*ref_img.size(), 1).repeat(1, 1, text_reshaped.shape[-1])#nx512x20
        feat_reshaped = torch.cat([text_reshaped, img_reshaped],dim=1)#nx1280x20
        feat_reshaped = self.all_conv(feat_reshaped).permute(0,2,1)#nx20x768

        att_text = attn * feat_reshaped #nx20x768
        g_text = self.W_v(att_text.permute(0,2,1)).permute(0,2,1)#nx20x768
        '''tmp'''
        att_text = text+g_text #nx20x768

        avg_text = torch.mean(att_text, dim=1)#nx768
        final_text = self.text_bn(avg_text)#nx768
#        final_text = self.dropout(final_text)
        final_text = self.text_fc(final_text)#nx512
        final_text = F.normalize(final_text) * 4

        return final_text


def create_models(configs, vocabulary):
    text_encoder = text_encoder_factory(vocabulary, config=configs)
    #text_encoder2 = text_encoder_factory(vocabulary, config=configs)
    lower_img_encoder, upper_img_encoder, image_classifier, stage2_encoder = image_encoder_factory(config=configs, pos='layer4')
    _, upper_img_encoder2, _, _= image_encoder_factory(config=configs, pos='layer2')
    _, upper_img_encoder3, _, _= image_encoder_factory(config=configs, pos='layer3')

    _, upper_img_encoder_t, _, _= image_encoder_factory(config=configs, pos='layer4')
    _, upper_img_encoder_t3, _, _= image_encoder_factory(config=configs, pos='layer3')

    layer_shapes = lower_img_encoder.layer_shapes()
    if configs['projector'] == 'linear':
        pass
    #    projector = correction(512, 512)
    if configs['projector'] == 'bilinear':
        projector = nn.Bilinear(configs['feature_size'], configs['text_feature_size'], configs['feature_size'])
    #elif configs['projector'] == 'mlp':
    #    projector = MLP(layer_shapes['layer4'], upper_img_encoder.feature_size)
    compositors = transformer_factory({'layer4': layer_shapes['layer4'],
                                       'image_feature_size': upper_img_encoder.feature_size,
                                       'text_feature_size': text_encoder.feature_size}, configs=configs)
    #compositors = {'layer4': nn.Bilinear(49, 1, 49)} #image text compose
    hidden_size = 768
    feature_size = configs['feature_size']

#    text_transformation = nn.Sequential(
#            nn.BatchNorm1d(hidden_size),
#            nn.Linear(hidden_size, feature_size),
#            nn.Dropout(p=0.2),
#        )
    text_transformation = text_transform(feature_size, hidden_size)
    #targ_fc = text_targ_fc(feature_size)
    #text_transformation = text_transform(layer_shapes['layer4'], hidden_size)

    compositors2 = transformer_factory({'layer4': layer_shapes['layer2'],
                                       'image_feature_size': upper_img_encoder2.feature_size,
                                       'text_feature_size': text_encoder.feature_size}, configs=configs)
    compositors3 = transformer_factory({'layer4': layer_shapes['layer3'],
                                       'image_feature_size': upper_img_encoder3.feature_size,
                                       'text_feature_size': text_encoder.feature_size}, configs=configs)
    text_transformation2 = text_transform(feature_size, hidden_size)
    text_transformation3 = text_transform(feature_size, hidden_size)
    models = {
        'text_encoder': text_encoder,
        'text_transformation': text_transformation,
        'text_transformation3': text_transformation3,
        'lower_image_encoder': lower_img_encoder,
        'upper_image_encoder': upper_img_encoder,
        'upper_image_encoder3': upper_img_encoder3,
        'upper_image_encoder_t': upper_img_encoder_t,
        'upper_image_encoder_t3': upper_img_encoder_t3,
        'layer4': compositors,
        'layer3': compositors3,
    }
#    models.update(compositors)

    if configs['num_gpu'] >= 1:
        for name, model in models.items():
            models[name] = GradientControlDataParallel(model.cuda())
            #models[name] = BalancedDataParallel(8, model.cuda())

    return models
