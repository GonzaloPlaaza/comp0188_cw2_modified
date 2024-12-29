from abc import abstractmethod
import torch.nn as nn
import torch
from typing import Dict

class BaseModel(nn.Module):

  def __init__(self):
      """Abstract base class for torch models asserting a required reset method
      reset should be overitten such that when called, model parameters are 
      re-initialised. The forward method should be overwritten in the same way
      all forward methods are defined when subclassing nn.Module classes
      """
      super().__init__()

  @abstractmethod
  def forward(self, *args, **kwargs):
      pass

  @abstractmethod
  def reset(self):
      pass