from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
import torch.nn as nn

def calc_kernel_output_size(
    w:int, k:int, p:int, d:int, s:int
    ):
    return np.floor(((w+2*p-d*(k-1)-1)/s)+1)

@dataclass
class ConvolutionLayersConfig:
    """Data class for defining CNN structures. CNN layers should be added to the 
    layers parameter. Calling get_output_dims or get_output_channels prints the 
    __output__ dimensions/channels of the layers 
    """
    input_dim:int
    input_channels:int
    layers:List[nn.Module]
    
    def get_output_dims(self):
        _output_dims = []
        _input_dim = self.input_dim
        _out_dim = _input_dim
        for i in self.layers:
            if isinstance(i,nn.ConvTranspose2d):
                assert len(i.kernel_size) == 2
                assert i.kernel_size[0] == i.kernel_size[1]
                assert i.stride[0] == i.stride[1]
                _out_dim = (
                    ((_input_dim-1)*i.stride[0]) - (2*i.padding[0])
                )
                _out_dim = _out_dim + i.dilation[0]*(i.kernel_size[0]-1)+i.output_padding[0]+1
                _input_dim = _out_dim
            elif isinstance(i,nn.Upsample):
                _out_dim = i.scale_factor*_out_dim
                _input_dim = _out_dim
            elif hasattr(i,"kernel_size"):
                # print(i)
                assert len(i.kernel_size) == 2
                assert i.kernel_size[0] == i.kernel_size[1]
                assert i.stride[0] == i.stride[1]
                if isinstance(i.padding, Tuple):
                    assert i.padding[0] == i.padding[1]
                    _padding = i.padding[0]
                else:
                    _padding = i.padding
                if isinstance(i.dilation, Tuple):
                    assert i.dilation[0] == i.dilation[1]
                    _dilation = i.dilation[0]
                else:
                    _dilation = i.dilation
                _out_dim = calc_kernel_output_size(
                    w=_input_dim,
                    k=i.kernel_size[0],
                    p=_padding,
                    d=_dilation,
                    s=i.stride[0]
                    )
                _input_dim = _out_dim
            else:
                _out_dim = _input_dim
            _output_dims.append(
                _out_dim
            )
        return _output_dims

    def get_output_channels(self):
        _output_channels = []
        _input_channels = self.input_channels
        for i in self.layers:
            if hasattr(i,"out_channels"):
                _output_channels.append(i.out_channels)
                _input_channels = i.out_channels
            else:
                _output_channels.append(_input_channels)
        return _output_channels