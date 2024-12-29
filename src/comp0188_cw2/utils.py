from jaxtyping import jaxtyped, Int,Float
from typeguard import typechecked as typechecker
import numpy as np
from typing import List, Dict
import re
from tqdm import tqdm 
import os
import wandb
import h5py
import pandas as pd
from io import StringIO


from .config import WANDB_PROJECT

@jaxtyped(typechecker=typechecker)
def to_greyscale(
    c_img:Int[np.ndarray, "batch_size row_pixels col_pixels filters"]
    ) -> Float[np.ndarray, "batch_size row_pixels col_pixels"]:
    """Converts numpy array of dimension: 
    (batch_size, height, width, filters), defining an RGB image to a greyscale 
    image of dimension: (batch_size, height, width)

    Returns:
        np.ndarray: Greyscale version of the input image
    """
    return np.matmul(c_img, [0.2989, 0.5870, 0.1140])

def load_all_files(
    h5_dir:str,
    file_pattern:str,
    keys:List[str]
    )->Dict[str,np.ndarray]:
    """Function to load all h5 files from a specified location into a numpy 
    array

    Args:
        h5_dir (str): Directory containing target h5 files
        file_pattern (str): A regex defining the names of the target h5 files 
        in the directory
        keys (List[str]): The keys to load from the h5 files

    Returns:
        Dict[str,np.ndarray]: Dictionary of numpy arrays
    """
    all_files = [
        i for i in os.listdir(h5_dir) if re.match(file_pattern,i) is not None
    ]
    all_arrays = {k:[] for k in keys}
    for f_name in tqdm(all_files):
        with h5py.File(
            os.path.join(h5_dir, f_name),
            'r') as h5f:
            for k in keys:
                all_arrays[k].append(h5f[k][()])
    for k in keys:
        all_arrays[k] = np.concatenate(all_arrays[k],axis=0)
    return all_arrays


def wandb_csv_to_pandas(
    file_name:str,
    run_name:str, 
    project_name:str=WANDB_PROJECT
    ):
    """Function to load a target csv file from weights and biases into a pandas 
    dataframe

    Args:
        file_name (str): File name in weights and biases
        run_name (str): Run name within weights and biases to which the csv 
        file is associated
        project_name (str, optional): Project name within weights and biases to 
        which the run is associated. Defaults to WANDB_PROJECT.

    Returns:
        _type_: _description_
    """
    api = wandb.Api()
    runs = api.runs(project_name)
    run = [r for r in runs if r.name == run_name][0]
    with run.file(file_name).download(replace=True) as f:
        csv_file = f.read()
    df = pd.read_csv(StringIO(csv_file), sep=",", header=None)
    return df
