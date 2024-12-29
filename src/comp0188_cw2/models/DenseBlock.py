import torch.nn as nn
import torch
from typing import Optional

class DenseBlock(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        batch_norm:bool,
        dropout:float,
        actvton:Optional[nn.Module] = None,
        bias=True,
        bias_init:int=0
        ):
        """torch Module defining a single dense fully connected layer.

        Args:
            input_dim (int): Input dimension of the fully connected layer
            output_dim (int): Output dimension of the fully connected layer
            batch_norm (bool): Boolean as to whether apply batch norm after 
            applying the activation function
            dropout (float): Proportion of dropout to apply after the batch 
            norm.
            actvton (nn.Module, optinal): Activation function to apply. 
            Defaults to None.
            bias (bool, optional): Boolean defining whether to add a bias term 
            to the Linear layer. Defaults to True.
            bias_init (int, optional): Value to initialise the bias term to. 
            Defaults to 0.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.actvton = actvton
        self.bias_init = bias_init

    def forward(self, input:torch.Tensor)->torch.Tensor:
        input = self.linear(input)
        if self.actvton is not None:
            input = self.actvton(input)
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input

    def reset(self,gain=1.0):
        nn.init.xavier_uniform_(self.linear.weight.data,gain=gain)
        self.linear.bias.data.fill_(self.bias_init)
