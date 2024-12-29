import torch
import torch.nn as nn
from typing import Dict

from ..Logging import logger
from .CNN import CNN
from .MLP import MLP
from .base import BaseModel

class JointCNNEncoder(BaseModel):
    
    def __init__(
        self, 
        cnn:CNN,
        dense:MLP
        ) -> None:
        """Torch module for jointly encoding two images into a single latent 
        space using a CNN and fully connected MLP.

        Args:
            cnn (CNN): CNN encoder
            dense (MLP): Fully connected encoder
        """
        super().__init__()
        self._cnn = cnn
        self._dense = dense
        self._flatten = nn.Flatten()
    
    def reset(self, gain:float=1.0):
        self._cnn.reset(gain=gain)
        self._dense.reset(gain=gain)
    
    def forward(
        self, 
        x:torch.Tensor,
        )->Dict[str,torch.Tensor]:
        _x = self._dense(self._flatten(self._cnn(x)))
        return _x


class JointCNNOnlyEncoder(BaseModel):

    def __init__(
        self, 
        cnn:CNN,
        ) -> None:
        super().__init__()
        self._cnn = cnn
        self._flatten = nn.Flatten()
    
    def reset(self, gain:float=1.0):
        self._cnn.reset(gain=gain)
    
    def forward(
        self, 
        x:torch.Tensor,
        )->Dict[str,torch.Tensor]:
        _x = self._flatten(self._cnn(x))
        return _x
