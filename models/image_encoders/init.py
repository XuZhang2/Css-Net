import torch.nn.init as init
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=1e-7)
        if m.bias != None:
            init.constant_(m.bias.data, 0.0)

def weights_init_classifier_less(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
