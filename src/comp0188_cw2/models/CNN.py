import torch.nn as nn
from typing import List 

from .CNNConfig import ConvolutionLayersConfig

class CNN(nn.Module):

    def __init__(
        self,
        convolution_config:ConvolutionLayersConfig,
        bias_init:int=0
        ):
        super().__init__()
        module_list = nn.ModuleList(
            convolution_config.layers
        )
        self.cnn_output_dim = convolution_config.get_output_dims()
        self.cnn_output_channels = convolution_config.get_output_channels()
        self.bias_init = bias_init
        self.module_list = module_list
        self.reset()

    def forward(self,x):
        _x = x
        for l in self.module_list:
            _x = l(_x)
        return _x

    def reset(self, gain:float=1.0):
        for layer in self.module_list:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(self.bias_init)
            else:
                pass