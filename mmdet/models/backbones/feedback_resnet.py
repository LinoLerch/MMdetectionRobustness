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
        self.up = nn.Upsample(scale_factor=32)
        self.conv = nn.Conv2d(2048, 3, kernel, padding=1)
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
        self.resnet = ResNet(**kwargs)
        self.fb_con = FBConnection()

    def forward(self, input):
        stage_outputs=[]
        b, c, size_y, size_x  = input.shape
        feedback = torch.zeros((1,3,size_y,size_x))
        for i in range(3):
            input = modulate(input, feedback)
            stage_outputs = self.resnet(input)
            feedback = self.fb_con(stage_outputs[-1])
        return tuple(stage_outputs)


def modulate(x, y):
    return(x * (1 + y))