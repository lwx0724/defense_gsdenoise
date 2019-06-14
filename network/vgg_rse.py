'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers.feat_noise import Noise
from config import  cfg
__all__ = [
    'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
]


cfg2 = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    #这是另一种不用对抗训练的方法,增加noise层，std比赛中设置到0.6
    def __init__(self,vgg_name,bn_bool=True,noise_init=0.1, noise_inner=0.1,num_classes=110, init_weights=True):
        super(VGG, self).__init__()
        self.noise_init = noise_init
        self.noise_inner = noise_inner
        self.features = self._make_layers(cfg2[vgg_name],bn_bool)
        self.input_size = [224,224,3]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self,cfg,batch_norm=False):
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i == 0:
                    noise_layer = Noise(self.noise_init)
                else:
                    noise_layer = Noise(self.noise_inner)

                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [noise_layer,conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]#比vgg16多了一层
                else:
                    layers += [noise_layer,conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('vgg16',bn_bool=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(cfg.pretrain))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('vgg16',bn_bool=True, **kwargs)
    if pretrained:
        print('pretrain_dir:',cfg.pretrain2)
        model.load_state_dict(torch.load(cfg.pretrain2)['model_state_dict'])
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('vgg19',bn_bool=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(cfg.pretrain))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('vgg19',bn_bool=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(cfg.pretrain))
    return model
