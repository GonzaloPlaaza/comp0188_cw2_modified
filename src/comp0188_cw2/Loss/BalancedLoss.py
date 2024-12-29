import torch
from typing import Dict, Optional

from ..Metric.WandBMetricOrchestrator import WandBMetricOrchestrator


class TrackerBalancedLoss:

    def __init__(
        self,
        loss_lkp:Dict,
        mo:Optional[WandBMetricOrchestrator]=None,
        name:Optional[str] = "",
        return_metric_value_dict:bool = False
        ):
        """Class for handling MultiLoss outputs

        Args:
            loss_lkp (Dict[str:torch.nn.modules.loss]): Dictionary containing 
            the required losses and lookup values. The lookup values should
            match those passed to the call method in act and pred parameters
        """
        self.loss_lkp = loss_lkp
        self.mo = mo
        self.__step = 1
        self.name = name
        self.return_metric_value_dict = return_metric_value_dict

    def __call__(
        self, 
        pred:Dict[str, torch.Tensor],
        act:Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        """Evaluates the input values against the loss functions specified in
        the init.

        Args:
            act (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": actual_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init
            pred (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": predicted_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init

        Returns:
            Dict[str:Any]: Dictionary of evaluated results. The keys will match
            those provided in the multi_loss parameter
        """
        loss = 0
        _metric_value_dict = {}
        for key in self.loss_lkp.keys():
            _loss = self.loss_lkp[key](pred[key], act[key])
            _metric_value_dict[f"{key}_{self.name}_loss"] = {
                "label":f"step_{self.__step}",
                "value":_loss
            }
            loss += _loss
        if self.mo is not None:
            self.mo.update_metrics(metric_value_dict=_metric_value_dict)
        out_loss = torch.mean(loss)
        self.__step += 1
        if self.return_metric_value_dict:
            return out_loss, _metric_value_dict
        
        else:
            return out_loss