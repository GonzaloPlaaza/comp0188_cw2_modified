
import torch
from typing import Dict, Tuple, Optional, Any
import logging
from tqdm import tqdm
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )
import numpy as np

class TrainSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True,
        beta_procedure:str="None",
        epochs:int = 10,
        n_cycle:int=1
        ) -> None:
        """Class which runs a single epoch of training.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds
        self.steps = 0
        self.epochs = epochs
        self.beta_procedure = beta_procedure    
        self.n_cycle = n_cycle
        
    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        mac:bool,
        optimizer:torch.optim.Optimizer,
        criterion:CriterionProtocol,
        logger:logging.Logger,
        return_metric_value_dict:bool = False,
        metric_value_dict:Optional[Dict[str,Any]]= None
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of training
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            optimizer (torch.optim.Optimizer): Torch optimiser to use in training
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            logger (logging.Logger): Logger object to use for printing to terminal
        Raises:
            RuntimeError: Captures generic runtime errors that may occur during 
            training

        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """
        losses = 0
        self.n_iters = self.epochs*len(data_loader)
        if return_metric_value_dict:
            metric_value_dict = {k:0 for k in metric_value_dict.keys()}

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
        model.train()
        
        preds = []
        betas = []
        range_gen = tqdm(
            enumerate(data_loader),
            total=len(data_loader)
            #desc=f"Epoch {int(epoch)}/{epochs}",
            )
        for i, vals in range_gen:

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
                input_vals = {val:Variable(input_vals[val]) for val in input_vals}
                output_vals = {val:Variable(output_vals[val])
                            for val in output_vals}

            optimizer.zero_grad()


            # Compute output
            if self.half_precision:
                with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                        if return_metric_value_dict:
                            train_loss, train_loss_dict = criterion(output, input_vals)
                        
                        else:
                            train_loss = criterion(output, output_vals)
                    
            else:
                output = model(**input_vals)
                if return_metric_value_dict:
                    if self.beta_procedure == "constant":
                        beta = criterion.loss_lkp['KL_div'].beta

                    elif self.beta_procedure == "monotonic":
                        beta = beta_monotonic(self.steps)

                    elif self.beta_procedure == "cycle_linear":
                        beta = beta_cycle_linear(self.n_iters, self.steps, start=0.0, stop=1.5,  n_cycle=self.n_cycle, ratio=0.5)
                    
                    betas.append(beta)
                    criterion.loss_lkp['KL_div'].beta = beta   
                    train_loss, train_loss_dict = criterion(output, input_vals)
                    for loss_type in train_loss_dict.keys():
                        
                        if "KL_div" in loss_type:
                            metric_value_dict[loss_type] += train_loss_dict[loss_type]["value"].detach().cpu().item()/(input_vals["images"].shape[0]*beta)

                        else:
                            metric_value_dict[loss_type] += train_loss_dict[loss_type]["value"].detach().cpu().item()/input_vals["images"].shape[0]

                else:
                    train_loss = criterion(output, output_vals)
            
            if self.cache_preds:
                preds.append({k:output[k].detach().cpu() for k in output.keys()})
            
            denom += 1
            self.steps += 1
            # losses.update(train_loss.data[0], g.size(0))
            # error_ratio.update(evaluation(output, target).data[0], g.size(0))

            try:
                # compute gradient and do SGD step
                if return_metric_value_dict:
                    train_loss = train_loss/input_vals["images"].shape[0]
                train_loss.backward()

                optimizer.step()
            except RuntimeError as e:
                logger.debug("Runtime error on training instance: {}".format(i))
                raise e
            
            losses += train_loss.detach().cpu()
            
        _prd_lst = {}

        if self.cache_preds:
            for k in preds[0].keys():
                _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)

        if return_metric_value_dict:
            losses = losses/denom    
            for loss_type in metric_value_dict.keys():
                metric_value_dict[loss_type] = metric_value_dict[loss_type]/denom
            return losses, metric_value_dict, _prd_lst, betas
        
        else:
            losses = losses/denom
            return losses, _prd_lst
        

        
    
class KL_divergence:
    
    def __init__(self, 
                beta: float = 1.0,
                batch_size: int = 128,
                ) -> None:
        self.beta = beta   
        self.batch_size = batch_size 
    
    def __call__(self, z):
        mu = z[0]
        logvar = z[1]
        beta_KL = -self.beta * 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return beta_KL

def beta_monotonic(iter):
    if iter < 995:
        return 0.001 + 0.999 * (iter/995)

    elif iter >= 995:
        return 1

def beta_cycle_linear(n_iters, iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    n_iters, iter, n_cycle = int(n_iters), int(iter), int(n_cycle)
    L = np.ones(n_iters) * stop
    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iters):
            L[int(i+c*period)] = v
            v += step
            i += 1
    if L[iter] < 0.0001:
        return 0.0001
    else:
        return L[iter]