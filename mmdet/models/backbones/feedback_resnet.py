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
    def __init__(self, scale_factor, kernel=3, activation=F.relu, neighborhood=5, alpha=0.0001):
        super(FBConnection, self).__init__()
        self.activation = activation
        self.up = nn.Upsample(scale_factor=scale_factor)
        in_channels=256*scale_factor
        self.conv = nn.Conv2d(in_channels, 64, kernel, padding=1)
        self.lrn = nn.LocalResponseNorm(neighborhood, alpha, 0.5, 1.0)
        
    def forward(self, xs):
        x = self.up(xs)
        x = self.conv(x)
        x = self.activation(x)
        x = self.lrn(x)
        return x

@BACKBONES.register_module()
class FeedbackResNet(nn.Module):

    def __init__(self, feedback_type, **kwargs):
        super(FeedbackResNet, self).__init__()
        if feedback_type not in ["mod","add"]:
            raise KeyError(f'invalid feedback type, available options: "mod","add"')
        
        self.resnet = ResNet(**kwargs)
        self.fb_type = feedback_type
        scale_factor = 8 if (self.fb_type=="mod") else 4
        self.fb_con = FBConnection(scale_factor)

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
            if self.fb_type=="mod":
                x = modulate(x, feedback)
                stage_outputs = _forward_stages(x)
                feedback = self.fb_con(stage_outputs[-1])
            elif self.fb_type=="add":
                x = x + feedback
                stage_outputs = _forward_stages(x)
                feedback = self.fb_con(stage_outputs[-2])

        return tuple(stage_outputs)

def modulate(x, y):
    return(x * (1 + y))