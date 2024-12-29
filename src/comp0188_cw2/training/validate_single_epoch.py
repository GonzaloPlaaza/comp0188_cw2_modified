
import torch
from typing import Tuple, Dict, Optional, Any
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )

class ValidateSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True,
        cache_inputs:bool=False,
        return_metric_value_dict:bool = False,
        metric_value_dict:Optional[Dict[str,Any]]= None
        ) -> None:
        """Class which runs a single epoch of validation.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds
        self.cache_inputs = cache_inputs
        self.return_metric_value_dict = return_metric_value_dict
        self.metric_value_dict = metric_value_dict

    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        mac:bool,
        criterion:CriterionProtocol
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of validation
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            
        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """

        losses = 0
        if self.return_metric_value_dict:
            metric_value_dict = {k:0 for k in self.metric_value_dict.keys()}

        denom = 0
        if gpu:
            if mac:
                _device = "mps"
            else:
                _device = "cuda"
        else:
            _device = "cpu"
            
        if self.half_precision:
            losses = losses.half()
            denom = denom.half()
        model.eval()
        preds = []
        true = []
        inputs = []
        with torch.no_grad():
            for i, vals in enumerate(data_loader):
                # Prepare input data
                input_vals = vals.input
                output_vals = vals.output
                if gpu:
                    if mac:
                        input_vals = {
                            val:input_vals[val].to(_device) for val in input_vals
                            }
                        output_vals = {
                            val:output_vals[val].to(_device) for val in output_vals
                            }
                    else:
                        input_vals = {
                            val:input_vals[val].cuda() for val in input_vals
                            }
                        output_vals = {
                            val:output_vals[val].cuda() for val in output_vals
                            }
                else:
                    input_vals = {
                        val:Variable(input_vals[val]) for val in input_vals
                        }
                    output_vals = {
                        val:Variable(output_vals[val]) for val in output_vals
                        }

                # Compute output
                if self.half_precision:
                    with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                else:
                    output = model(**input_vals)

                # Logs
                if self.return_metric_value_dict:
                    val_loss, val_loss_dict = criterion(output, input_vals)
                    for key in val_loss_dict.keys():
                        if "KL_div" in key:
                            metric_value_dict[key] += val_loss_dict[key]["value"].detach().cpu().item()/(input_vals["images"].shape[0]*criterion.loss_lkp["KL_div"].beta)
                        metric_value_dict[key] += val_loss_dict[key]["value"].detach().cpu().item()/input_vals["images"].shape[0]
                
                else:
                    val_loss = criterion(output, output_vals)
                
                if self.return_metric_value_dict:
                    losses += val_loss.detach().cpu()/input_vals["images"].shape[0]
                else:
                    losses += val_loss.detach().cpu()
                denom += 1
                
                if self.cache_preds:

                    if self.return_metric_value_dict:
                        output["mu"] = output["KL_div"][0]
                        output["logvar"] = output["KL_div"][1]
                        output.pop("KL_div")

                    else:
                        true.append({k:output_vals[k].detach().cpu() for k in output_vals.keys()})
                    
                    preds.append({k:output[k].detach().cpu() for k in output.keys()})

                if self.cache_inputs:
                    inputs.append({k:input_vals[k].detach().cpu() for k in input_vals.keys()})

        _prd_lst = {}
        _true_lst = {}
        _input_lst = {}
        if self.cache_preds:

            if self.return_metric_value_dict:
                for k in preds[0].keys():
                    i = 0
                    prd_list = []
                    while i < 10:
                        prd_list.append(preds[i][k])
                        i += 1
                    _prd_lst[k] = torch.concat(prd_list,dim=0)

            else: 
                for k in preds[0].keys():
                    _true_lst[k] = torch.concat([t[k] for t in true],dim=0)
                    _prd_lst[k] = torch.concat([p[k] for p in preds],dim=0)
        
        if self.cache_inputs:

            if self.return_metric_value_dict:
                for k in inputs[0].keys():
                    i = 0
                    _input_list = []
                    while i < 10:
                        _input_list.append(inputs[i][k])
                        i += 1
                    _input_lst[k] = torch.concat(_input_list,dim=0)
            
            else:
                for k in inputs[0].keys():
                    _input_lst[k] = torch.concat([t[k] for t in inputs],dim=0)
        
        if self.return_metric_value_dict:
            losses = losses/denom
            for key in metric_value_dict.keys():
                metric_value_dict[key] = metric_value_dict[key]/denom
            return losses, metric_value_dict, _prd_lst, _true_lst, _input_lst
        
        else:
            losses = losses/denom
            return losses, _prd_lst, _true_lst, _input_lst