import torch
import torch.nn as nn
from typing import List, Optional

from .DenseBlock import DenseBlock
from .base import BaseModel

class MLP(BaseModel):
    def __init__(
        self,
        input_dim:int,
        hidden_dims:List[int],
        output_dim:int,
        bias_init:int=0,
        actvton:Optional[nn.Module] = None,
        fnl_act:Optional[nn.Module] = None,
        fnl_dropout:Optional[float] = None,  
        fnl_bn:Optional[bool] = False,  
        batch_norms:Optional[List[bool]] = None,
        dropouts:Optional[List[float]] = None
        ):
        """Torch module defining a fully connected feed forward neural network.
        If the input layer is 4, output layer is 1 and the hidden layer is 6, 
        the network will compose of 2 fully connected layers:
        - Layer 1: input=4, output=6
        - Layer 2: input=6, output=1

        Args:
            input_dim (int): Input dimension of the first feedforward layer
            hidden_dims (List[int]): Dimensions of the hidden layers
            output_dim (int): Output dimension of the final feedforward layer
            bias_init (int, optional): Value to initialise the bias term to. 
            Defaults to 0.
            actvton (nn.Module, optinal): Activation function to apply to 
            to hidden layers. 
            fnl_act (Optional[nn.Module], optional): The activation to apply 
            to the final hidden layer. If None, no activation function is 
            applied. Defaults to None.
            batch_norms (Optional[List[bool]], optional): List of booleans 
            defining whether to apply batchnorm after each hidde layer. 
            If not None, list must be the same length as hidden_dims
            Defaults to None.
            dropouts (Optional[List[float]], optional): List of dropout 
            proportions to apply after each hidden layer. 
            If not None, list must be the same length as hidden_dims
            Defaults to None.
        """
        super().__init__()
        __dims = [input_dim, *hidden_dims]
        if batch_norms is None:
            batch_norms = [False]*len(hidden_dims)
        assert len(batch_norms) == len(hidden_dims)    
        if dropouts is None:
            dropouts = [0]*len(hidden_dims)
        assert len(dropouts) == len(hidden_dims)
        self.module_lst = nn.ModuleList()
        if len(hidden_dims) > 0:
            for idx in range(len(__dims)-1):
                self.module_lst.append(
                    DenseBlock(
                        input_dim=__dims[idx],
                        output_dim=__dims[idx+1],
                        actvton=actvton,
                        dropout=dropouts[idx],
                        batch_norm=batch_norms[idx],
                        bias_init=bias_init
                    )
                    )
        self.bias_init = bias_init
        self.module_lst.append(
            nn.Linear(
                __dims[-1],
                output_dim
            )
        )
        if fnl_bn:
            self.module_lst.append(nn.BatchNorm1d(output_dim))
        if fnl_act is not None:
            self.module_lst.append(fnl_act)
        if fnl_dropout is not None:
            self.module_lst.append(nn.Dropout(fnl_dropout))
        

        self.reset()

    def forward(self,x:torch.Tensor):
        _x = x
        for layer in self.module_lst:
            _x = layer(_x)
        return _x

    def reset(self, gain:float=1.0):
        for layer in self.module_lst:
            if isinstance(layer, DenseBlock):
                layer.reset(gain=gain)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(self.bias_init)
            else:
                pass