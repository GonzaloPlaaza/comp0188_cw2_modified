import numpy as np
from typing import List, Dict, Callable, Optional
from torch.utils.data import Dataset

from pymlrf.types import DatasetOutput

class NpDictDataset(Dataset):

    def __init__(
        self,
        array_dict:Dict[str,np.ndarray],
        dep_vars:List[str],
        indep_vars:List[str],
        transform_lkp:Optional[Dict[str,Callable]] = None,
        target_offset:int=0
        ):
        """torch Dataset class taking as input a dictionary of arrays, relevant
        transformations to apply to the arrays and an optional parameter for 
        defining the prediction target to be n steps a ahead i.e., predicting 
        $y_{t+target_offset}$ with $x_{t}$.

        Args:
            array_dict (Dict[str,np.ndarray]): A dictionary of arrays where each 
            key corrosponds to either a dependant or independant variable 
            defined in dep_vars or indep_vars
            dep_vars (List[str]): List of dependent variables i.e. the values to 
            predict
            indep_vars (List[str]): List of independant variables i.e., the 
            inputs to the model
            transform_lkp (Optional[Dict[str,Callable]], optional): 
            A dictionary of preprocessing steps to apply to each input/output. 
            Defaults to None.
            target_offset (int, optional): The number of timesteps in the future 
            to define the dependent variable. Defaults to 0.
        """
        self.__dep_vars = dep_vars
        self.__indep_vars = indep_vars
        self.__array_dict = array_dict
        _msg = """
        The array_dict must contain an array defining the terminal steps of trajectories where 1 defines the terminal step.
        This must be assigned the key 'terminals'
        """
        assert "terminals" in array_dict.keys(), _msg
        valid_idx = np.arange(array_dict["terminals"].shape[0])
        if self.__array_dict["terminals"][-1] != 1:
            self.__array_dict["terminals"][-1] = 1
        self._target_offset = target_offset
        _idx_to_drop = []
        if target_offset > 0:
            for i in range(target_offset):
                _idx_to_drop.append(
                    np.where(array_dict["terminals"]==1)[0] - i 
                    )
            _idx_to_drop = np.concatenate(_idx_to_drop, axis=0)
            _shape_pre = valid_idx.shape[0]
            valid_idx = valid_idx[~np.isin(valid_idx,_idx_to_drop)]
            _to_drp = target_offset*(array_dict["terminals"]==1).sum()
            _drped = _shape_pre - valid_idx.shape[0]
            assert _to_drp == _drped, f"{_to_drp}, {_drped}"
        self._valid_idx = valid_idx
        self.__length = len(self._valid_idx)
        
        if not transform_lkp:
            self.transform_lkp = {}
            for k in [*self.__dep_vars, *self.__indep_vars]:
                self.transform_lkp[k] = lambda x: x
        else:
            self.transform_lkp = transform_lkp
        _tst = set([*self.__dep_vars, *self.__indep_vars]) - set(self.transform_lkp.keys())
        assert len(_tst) == 0

    def __getitem__(self, idx:int)->DatasetOutput:
        idx = self._valid_idx[idx]
        _input_dict = {}
        _output_dict = {}
        for k in self.__indep_vars:
            _input_dict[k] = self.transform_lkp[k](self.__array_dict[k][idx,:])
        for k in self.__dep_vars:
            _output_dict[k] = self.transform_lkp[k](
                self.__array_dict[k][idx+self._target_offset,:]
                )
        return DatasetOutput(input=_input_dict,output=_output_dict)

    def __len__(self)->int:
        return self.__length
    
    def get_array_dict(self)->Dict[str,np.ndarray]:
        return self.__array_dict  
    
    def get_dep_vars(self)->List[str]:
        return self.__dep_vars
    
    def get_indep_vars(self)->List[str]:
        return self.__indep_vars
    