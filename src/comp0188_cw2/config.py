import os
from pymlrf.FileSystem import DirectoryHandler
from . import project_options

WANDB_PROJECT = "cw2_v2"

if project_options.collab:
    ROOT_PATH = "/content/drive/MyDrive/comp0188_2425/cw2"    
else:
    ROOT_PATH = "data/all_play_data_diverse"

if not os.path.exists(ROOT_PATH):
    raise FileNotFoundError(f"ROOT_PATH: {ROOT_PATH} does not exist")
    
FILE_PATH = os.path.join(ROOT_PATH, "all_play_data_diverse.h5")

_train_dh = DirectoryHandler(
    loc=os.path.join(ROOT_PATH,"train")
    )

_val_dh = DirectoryHandler(
    loc=os.path.join(ROOT_PATH,"val")
    )

_test_dh = DirectoryHandler(
    loc=os.path.join(ROOT_PATH,"test")
    )

debug_dh = DirectoryHandler(
    loc=os.path.join(ROOT_PATH, "debug")
)

_debug_train_dh = DirectoryHandler(
    loc=os.path.join(debug_dh.loc,"train")
    )

_debug_val_dh = DirectoryHandler(
    loc=os.path.join(debug_dh.loc,"val")
    )

_debug_test_dh = DirectoryHandler(
    loc=os.path.join(debug_dh.loc,"test")
    )

if project_options.debug:
    train_dh = _debug_train_dh
    val_dh = _debug_val_dh
    test_dh = _debug_test_dh
else:
    train_dh = _train_dh
    val_dh = _val_dh
    test_dh = _test_dh

if not train_dh.is_created:
    train_dh.create()
    
if not val_dh.is_created:
    val_dh.create()
    
if not test_dh.is_created:
    test_dh.create()