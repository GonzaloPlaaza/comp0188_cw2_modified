import os
import numpy as np
import wandb
import torch
from typing import Optional, Dict, Literal, Any, Tuple
from torch.utils.data import DataLoader
import logging
from pymlrf.utils import set_seed
from pymlrf.types import (
  CriterionProtocol, TrainSingleEpochProtocol, ValidateSingleEpochProtocol, 
  )
import pickle
from pymlrf.FileSystem import DirectoryHandler
torch.autograd.set_detect_anomaly(True)

from ..Metric.WandBMetricOrchestrator import WandBMetricOrchestrator
from ..models.base import BaseModel
from .train_single_epoch import TrainSingleEpoch
from .validate_single_epoch import ValidateSingleEpoch


def train(
    model:torch.nn.Module,
    train_data_loader:DataLoader,
    val_data_loader:DataLoader,
    gpu:bool,
    mac:bool,
    optimizer:torch.optim.Optimizer,
    criterion:CriterionProtocol,
    epochs:int,
    logger:logging.Logger,
    save_dir:str,
    scheduler:torch.optim.lr_scheduler.LRScheduler=None,
    train_epoch_func:TrainSingleEpochProtocol = TrainSingleEpoch(),
    val_epoch_func:ValidateSingleEpochProtocol = ValidateSingleEpoch(),
    seed: int = None,
    mo: WandBMetricOrchestrator = WandBMetricOrchestrator(),
    val_criterion:Optional[CriterionProtocol] = None,
    preds_save_type:Optional[Literal["pickle","csv"]] = None,
    output_dir:Optional[str] = None,
    return_metric_value_dict:bool = False,
    metric_value_dict_train:Optional[Dict[str,Any]] = None, 
    metric_value_dict_val:Optional[Dict[str,Any]] = None,
    beta_procedure:Optional[str] = None
    ) -> Tuple[WandBMetricOrchestrator,int]:
    """Function to run training and validation specified by the objects 
    assigned to train_epoch_func and val_epoch_func.

    Args:
        model (BaseModel): Torch model of type BaseModel i.e., it should
        subclass the BaseModel class
        train_data_loader (DataLoader): Torch data loader object
        val_data_loader (DataLoader): Torch data loader object
        gpu (bool): Boolean defining whether to use a GPU if available
        optimizer (torch.optim.Optimizer): Torch optimiser to use in training
        criterion (CriterionProtocol): Criterian to use for training or 
        for training and validation if val_criterion is not specified. I.e., 
        this could be nn.MSELoss() 
        epochs (int): Number of epochs to train for
        logger (logging.Logger): Logger object to use for printing to terminal
        save_dir (str): Directory where model checkpoints should be saved
        mo (WandBMetricOrchestrator, optional): Metric Orchestrator to use. 
        This abstracts logging to weights and biases away from the training 
        loop. Defaults to None.
        val_criterion (CriterionProtocol, optional): Criterian to use for 
        validation. If this is None, then the same criterion for training 
        will be used. Defaults to None.
        preds_save_type (Optional[Literal[&quot;pkl&quot;,&quot;csv&quot;]], optional): 
        Save format for predictions. If None, no predictions will be saved. 
        Defaults to None.
        output_dir (Optional[str], optional): Location to save model 
        predictions. Defaults to None.
        train_epoch_func (TrainSingleEpochProtocol, optional): Object which runs
        a single epoch of training. Defaults to TrainSingleEpoch().
        val_epoch_func (ValidateSingleEpochProtocol, optional): Object which runs
        a single epoch of validation. Defaults to ValidateSingleEpoch().
        seed (int, optional): Random seed to use. Defaults to None.

    Raises:
        ValueError: Value error raised if prediction save type is invalid

    Returns:
        WandBMetricOrchestrator: Weights and biases orchestrator object
        int: final epoch run
    """

    if seed is not None:
        set_seed(n=seed)

    if preds_save_type is not None:
        assert output_dir is not None

    mo.add_metric(
        nm="epoch_train_loss",
        rll_trans={}
        )
    mo.add_metric(
        nm="epoch_val_loss",
        rll_trans={}
        )
    
    logger.info("Running epochs: {}".format(epochs))
    # Add model to cuda device
    if gpu:
        if mac:
            model.to("mps")
        else:
            model.cuda()

    if val_criterion is None:
        val_criterion = criterion

    train_losses = []
    val_losses = []
    betas_total = []
    for epoch in np.arange(1,epochs+1):


        logger.info("Running training epoch")
        if return_metric_value_dict:
            train_loss_val, train_loss_dict, train_preds, betas =  train_epoch_func(
                model=model, data_loader=train_data_loader, gpu=gpu, mac=mac,
                optimizer=optimizer, criterion=criterion,logger=logger, 
                return_metric_value_dict=return_metric_value_dict, metric_value_dict=metric_value_dict_train)
            beta_epoch = betas[-1]
            #Concatenate betas_total and the new betas vector
            betas_total = np.concatenate([betas_total, betas])

        else:
            train_loss_val, train_preds =  train_epoch_func(
                model=model, data_loader=train_data_loader, gpu=gpu, mac=mac,
                optimizer=optimizer, criterion=criterion,logger=logger, return_metric_value_dict=return_metric_value_dict, metric_value_dict=metric_value_dict_train )
        epoch_train_loss = train_loss_val.numpy()

        logger.info("epoch {}\t training loss : {}".format(
                epoch, epoch_train_loss))
        
        if return_metric_value_dict:
            val_criterion.loss_lkp["KL_div"].beta = beta_epoch
            val_loss_val, val_loss_dict, val_preds, val_true, val_inputs = val_epoch_func(
                model=model, data_loader=val_data_loader, gpu=gpu, mac=mac,
                criterion=val_criterion)
        
        else:
            val_loss_val, val_preds, val_true, val_inputs = val_epoch_func(
                model=model, data_loader=val_data_loader, gpu=gpu, mac=mac,
                criterion=val_criterion)

        epoch_val_loss = val_loss_val.numpy()
        
        logger.info("Running validation")
        logger.info("epoch {}\t validation loss : {} ".format(
                epoch, epoch_val_loss))

        mo.update_metrics(metric_value_dict={
            "epoch_train_loss":{"label":"epoch_{}".format(epoch),
                                "value":epoch_train_loss},
            "epoch_val_loss":{"label":"epoch_{}".format(epoch),
                            "value":epoch_val_loss}
        })
        
        if return_metric_value_dict:
            mo.update_metrics_direct(metric_value_dict=train_loss_dict)
            mo.update_metrics_direct(metric_value_dict=val_loss_dict)
            mo.update_metrics_direct(metric_value_dict={"beta":betas})

        if scheduler:
            scheduler.step()

        chkp_pth = os.path.join(save_dir, "mdl_chkpnt_epoch_{}.pt".format(
            epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_train_loss,
            }, chkp_pth)

        if preds_save_type is not None:
            if preds_save_type == "pickle":
                for k in train_preds.keys():
                    with open(
                        os.path.join(
                            output_dir,
                            f"epoch_{epoch}_train_preds_{k}.pkl"
                            ), "wb"
                        ) as file:
                        pickle.dump(train_preds[k], file)
                for k in val_preds.keys():
                    with open(
                        os.path.join(
                            output_dir,
                            f"epoch_{epoch}_val_preds_{k}.pkl"
                            ), "wb"
                        ) as file:
                        pickle.dump(val_preds[k], file)

            elif preds_save_type == "csv":
                for k in train_preds.keys():
                    np.savetxt(
                        os.path.join(
                            output_dir,
                            f"epoch_{epoch}_train_preds_{k}.csv"
                            ),
                        train_preds[k].detach().cpu().float().numpy(),
                        delimiter=","
                        )
                for k in val_preds.keys():
                    np.savetxt(
                        os.path.join(
                            output_dir,
                            f"epoch_{epoch}_val_preds_{k}.csv"
                            ),
                        val_preds[k].detach().cpu().float().numpy(),
                        delimiter=","
                        )
            else:
                raise ValueError(
                    "preds_save_type must be either None, csv or pickle"
                    )
            
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if return_metric_value_dict:
            for key in val_loss_dict.keys():
                metric_value_dict_val[key].append(val_loss_dict[key])
                
            
            for key in train_loss_dict.keys():
                metric_value_dict_train[key].append(train_loss_dict[key])

    
    if return_metric_value_dict:
        metric_value_dict_train['total_loss'] = train_losses
        metric_value_dict_val['total_loss'] = val_losses
        return mo, epoch, metric_value_dict_train, metric_value_dict_val, betas_total

    else:
        return mo, epoch, train_losses, val_losses


class TorchTrainingLoop:

  def __init__(
        self,
        model:BaseModel,
        gpu:bool,
        mac:bool,
        optimizer:torch.optim.Optimizer,
        criterion:CriterionProtocol,
        epochs:int,
        logger:logging.Logger,
        scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        mo: WandBMetricOrchestrator=None,
        val_criterion:CriterionProtocol = None,
        preds_save_type:Optional[Literal["pkl","csv"]] = None,
        half_precision:bool=False,
        output_dir:Optional[str] = None,
        return_metric_value_dict:bool = False,
        metric_value_dict_train:Optional[Dict[str,Any]] = None,
        metric_value_dict_val:Optional[Dict[str,Any]] = None,
        annealing_beta_procedure:Optional[str] = None,
        n_cycle:Optional[int] = None
      ):
      """Class to orchestrate training and validating a model. CriterionProtocol
      if defined here 
      https://github.com/joshuaspear/pymlrf/blob/master/src/pymlrf/types.py
      and just ensures that the criterion is a callable function.

      Args:
          model (BaseModel): Torch model of type BaseModel i.e., it should
          subclass the BaseModel class
          gpu (bool): Boolean defining whether to use a GPU if available
          optimizer (torch.optim.Optimizer): Torch optimiser to use in training
          criterion (CriterionProtocol): Criterian to use for training or 
          for training and validation if val_criterion is not specified. I.e., 
          this could be nn.MSELoss() 
          epochs (int): Number of epochs to train for
          logger (logging.Logger): Logger object to use for printing to terminal
          mo (WandBMetricOrchestrator, optional): Metric Orchestrator to use. 
          This abstracts logging to weights and biases away from the training 
          loop. Defaults to None.
          val_criterion (CriterionProtocol, optional): Criterian to use for 
          validation. If this is None, then the same criterion for training 
          will be used. Defaults to None.
          preds_save_type (Optional[Literal[&quot;pkl&quot;,&quot;csv&quot;]], optional): 
          Save format for predictions. If None, no predictions will be saved. 
          Defaults to None.
          half_precision (bool, optional): Boolean defining whether to use 
          half-precision during training. Defaults to False.
          output_dir (Optional[str], optional): Location to save model 
          predictions. Defaults to None.
      """
      self.model = model
      self.optimizer = optimizer
      self.criterion = criterion
      self.epochs = epochs
      self.logger = logger
      self.scheduler = scheduler
      self.mo = mo
      self.val_criterion = val_criterion
      self.gpu = False
      self.mac = mac
      self.preds_save_type = preds_save_type
      self.half_precision = half_precision
      self.output_dir = output_dir
      self.cache_preds = False
      self.return_metric_value_dict = return_metric_value_dict
      self.metric_value_dict_train = metric_value_dict_train
      self.metric_value_dict_val = metric_value_dict_val
      self.beta_procedure = annealing_beta_procedure
      self.n_cycle = n_cycle

      assert self.beta_procedure in ["monotonic", "cycle_linear", "constant", None], "Invalid beta procedure"
      
      if preds_save_type is not None:
          self.cache_preds = True
      if gpu:
          if self.mac:
            if torch.backends.mps.is_available():
                self.gpu = True
            else:
                logger.warning("MPS device not available")
          else:
            if torch.cuda.is_available():
                self.gpu = True
            else:
                logger.warning("CUDA device not available")

  def training_loop(
        self,
        train_loader:DataLoader,
        val_loader:DataLoader,
        wandb_proj:str="",
        wandb_config:Dict={},
        wandb_name:Optional[str] = None,
        wandb_grp:Optional[str] = None,
        reset_kwargs:Optional[Dict[str,Any]] = {},
        reset_model:bool = True
        ) -> WandBMetricOrchestrator:
      wandb.init(
          project=wandb_proj,
          config=wandb_config,
          group=wandb_grp,
          name=wandb_name
          )
      torch.manual_seed(1)
      if reset_model:
        self.model.reset(**reset_kwargs)
      wandb.watch(self.model, log='all')
      chkpnt_dh = DirectoryHandler(
          loc=os.path.join(wandb.run.dir, "agent_checkpoints")
      )
      wandb_run_dir = wandb.run.dir
      if not chkpnt_dh.is_created:
          chkpnt_dh.create()

      if self.return_metric_value_dict:
          mo, epoch, train_loss_dict, val_loss_dict, betas_total = train(
                    model=self.model,
                    train_data_loader=train_loader,
                    val_data_loader=val_loader,
                    gpu=self.gpu,
                    mac=self.mac,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    epochs=self.epochs,
                    logger=self.logger,
                    save_dir=chkpnt_dh.loc,
                    scheduler=self.scheduler,
                    seed = 1,
                    mo = self.mo,
                    val_criterion = self.val_criterion,
                    train_epoch_func = TrainSingleEpoch(
                    half_precision=self.half_precision,
                    cache_preds=self.cache_preds,
                    beta_procedure = self.beta_procedure,
                    epochs = self.epochs,
                    n_cycle=self.n_cycle
                    ),
                    val_epoch_func = ValidateSingleEpoch(
                    half_precision=self.half_precision,
                    cache_preds=self.cache_preds,
                    return_metric_value_dict = self.return_metric_value_dict,
                    metric_value_dict=self.metric_value_dict_val
                    ),
                    preds_save_type = self.preds_save_type,
                    output_dir=self.output_dir,
                    return_metric_value_dict = self.return_metric_value_dict,
                    metric_value_dict_train=self.metric_value_dict_train,
                    metric_value_dict_val=self.metric_value_dict_val,
                    beta_procedure = self.beta_procedure
                )
      else:
            mo, epoch, train_losses, val_losses = train(
                        model=self.model,
                        train_data_loader=train_loader,
                        val_data_loader=val_loader,
                        gpu=self.gpu,
                        mac=self.mac,
                        optimizer=self.optimizer,
                        criterion=self.criterion,
                        epochs=self.epochs,
                        logger=self.logger,
                        save_dir=chkpnt_dh.loc,
                        scheduler=self.scheduler,
                        seed = 1,
                        mo = self.mo,
                        val_criterion = self.val_criterion,
                        train_epoch_func = TrainSingleEpoch(
                        half_precision=self.half_precision,
                        cache_preds=self.cache_preds
                        ),
                        val_epoch_func = ValidateSingleEpoch(
                        half_precision=self.half_precision,
                        cache_preds=self.cache_preds
                        ),
                        preds_save_type = self.preds_save_type,
                        output_dir=self.output_dir,
                        return_metric_value_dict = self.return_metric_value_dict,
                        metric_value_dict_train=self.metric_value_dict_train,
                        metric_value_dict_val=self.metric_value_dict_val)
          
      chckpnt_files = [f for f in os.listdir(chkpnt_dh.loc) if f[-3:]==".pt"]
      for i in chckpnt_files:
        artifact = wandb.Artifact(
            name=f"{wandb_name}-{i}", type='checkpoint'
            )
        artifact.add_file(os.path.join(chkpnt_dh.loc, i))
        wandb.log_artifact(artifact)
      wandb.finish()

      if self.return_metric_value_dict:
          return mo, epoch, train_loss_dict, val_loss_dict, wandb_run_dir, betas_total
      else:
        return mo, train_losses, val_losses, wandb_run_dir