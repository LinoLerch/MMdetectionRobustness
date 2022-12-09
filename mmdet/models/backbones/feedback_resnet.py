import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from .resnet import Bottleneck, ResNet

class FBConnection(nn.Module):
    """ based on: code from paper
    Feedback stream of the network
    upsampling of the feedforward stream, followed by a convolution and an
    activation function.
    """
    def __init__(self, kernel=3, activation=F.relu, neighborhood=5, alpha=0.0001):
        super(FBConnection, self).__init__()
        self.activation = activation
        self.up = nn.Upsample(scale_factor=8)
        self.conv = nn.Conv2d(2048, 64, kernel, padding=1)
        self.lrn = nn.LocalResponseNorm(neighborhood, alpha, 0.5, 1.0)
        
    def forward(self, xs):
        x = self.up(xs)
        x = self.conv(x)
        x = self.lrn(self.activation(x))
        return x

@BACKBONES.register_module()
class FeedbackResNet(nn.Module):

    def __init__(self, **kwargs):
        super(FeedbackResNet, self).__init__()
        # resnet = ResNet(**kwargs)
        # resnet_layers = nn.ModuleList(resnet.children())
        # self.stem = resnet_layers[:4]
        # self.resnet = resnet_layers[5:]
        self.resnet = ResNet(**kwargs)
        self.fb_con = FBConnection()

    def forward(self, x):
        # stem layer 1x
        x = self.resnet.conv1(x)
        x = self.resnet.norm1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        def _forward_stages(x):
            outs = []
            for i, layer_name in enumerate(self.resnet.res_layers):
                res_layer = getattr(self.resnet, layer_name)
                x = res_layer(x)
                if i in self.resnet.out_indices:
                    outs.append(x)
            return tuple(outs)

        # Stages 1-4 3x with feedback
        stage_outputs=[]
        feedback = torch.zeros(x.size(), device=x.device)
        for i in range(3):
            x = modulate(x, feedback)
            stage_outputs = _forward_stages(x)
            feedback = self.fb_con(stage_outputs[-1])
        return tuple(stage_outputs)


def modulate(x, y):
    return(x * (1 + y))