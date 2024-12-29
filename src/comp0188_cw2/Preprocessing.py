import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import h5py
from comp0188_cw2.config import (
    FILE_PATH, ROOT_PATH,
    train_dh, val_dh, test_dh
    )
from . import project_options
from comp0188_cw2.utils import to_greyscale
import numpy as np

def main():
    if project_options.debug:
        print("Generating debug datasets")
    tran_df = pd.read_csv(os.path.join(ROOT_PATH, "transition_df.csv"))
    tran_df_sub = tran_df[tran_df["episode_length"] < 75]
    print(tran_df_sub.shape)
    print(f"Transitions excluded: {(tran_df.shape[0]-tran_df_sub.shape[0])/tran_df.shape[0]}")

    _train_idx, test_idx = train_test_split(
        tran_df_sub["episode"].unique(),
        test_size=0.2,
        random_state=1
        )


    train_idx, val_idx = train_test_split(
        _train_idx,
        test_size=0.1,
        random_state=1
        )

    tran_df_sub["is_test"] = np.where(
        tran_df_sub["episode"].isin(test_idx),
        1,
        0
    )
    tran_df_sub["is_val"] = np.where(
        tran_df_sub["episode"].isin(val_idx),
        1,
        0
    )
    tran_df_sub["is_train"] = (1-tran_df_sub[["is_test","is_val"]].max(axis=1))
    n_train_tran = tran_df_sub["is_train"].sum()
    n_val_tran = tran_df_sub["is_val"].sum()
    n_test_tran = tran_df_sub["is_test"].sum()
    print(n_train_tran)
    print(n_val_tran)
    print(n_test_tran)
    assert tran_df_sub.shape[0] == (n_train_tran+n_val_tran+n_test_tran)

    _unique_ep = tran_df_sub[["episode","is_train", "is_val", "is_test"]].drop_duplicates()
    n_train_ep = _unique_ep["is_train"].sum()
    n_val_ep = _unique_ep["is_val"].sum()
    n_test_ep = _unique_ep["is_test"].sum()
    print(n_train_ep)
    print(n_val_ep)
    print(n_test_ep)
    assert _unique_ep.shape[0] == (n_train_ep+n_val_ep+n_test_ep)

    h5_file = h5py.File(FILE_PATH, 'r')

    REQ_KEYS = [
        'actions', 
        'front_cam_ob', 
        "terminals", 
        "mount_cam_ob",
        "ee_cartesian_pos_ob",
        "ee_cartesian_vel_ob",
        "joint_pos_ob"
        ]

    if not train_dh.is_created:
        train_dh.create()

    if not val_dh.is_created:
        val_dh.create()

    if not test_dh.is_created:
        test_dh.create()

    process_data_lkp = {
        "front_cam_ob": lambda x: to_greyscale(x.astype(int)).astype(np.float16),
        "mount_cam_ob": lambda x: to_greyscale(x.astype(int)).astype(np.float16),
        "actions": lambda x: x.astype(np.float16),
        "terminals": lambda x: x,
        "ee_cartesian_pos_ob": lambda x: x.astype(np.float16),
        "ee_cartesian_vel_ob": lambda x: x.astype(np.float16),
        "joint_pos_ob": lambda x: x[:,-2:].astype(np.float16)
    }

    train_index_chunks = np.array_split(
        tran_df_sub[tran_df_sub["is_train"] == 1].index, 20
        )
    if project_options.debug:
        train_index_chunks = [train_index_chunks[0]]
    no_train_pts_test = all(
        [re.match("train_[0-9]+.h5",i) is None for i in os.listdir(train_dh.loc)]
        )
    if no_train_pts_test:
        for idx, chunk in enumerate(train_index_chunks):
            print(f"Chunk: {idx+1}")
            with h5py.File(os.path.join(train_dh.loc, f'train_{idx}.h5'),'w') as h5f:
                for k in REQ_KEYS:
                    print(k)
                    h5f.create_dataset(
                        k,
                        data=process_data_lkp[k](
                            h5_file[k][chunk]
                            )
                        )

    val_index_chunks = np.array_split(
        tran_df_sub[tran_df_sub["is_val"] == 1].index, 5
        )
    if project_options.debug:
        val_index_chunks = [val_index_chunks[0]]
    no_val_pts_test = all(
        [re.match("val_[0-9]+.h5",i) is None for i in os.listdir(val_dh.loc)]
        )
    if no_val_pts_test:
        for idx, chunk in enumerate(val_index_chunks):
            print(f"Chunk: {idx+1}")
            with h5py.File(os.path.join(val_dh.loc, f'val_{idx}.h5'),'w') as h5f:
                for k in REQ_KEYS:
                    print(k)
                    h5f.create_dataset(
                        k,
                        data=process_data_lkp[k](
                            h5_file[k][chunk]
                            )
                        )


    test_index_chunks = np.array_split(
        tran_df_sub[tran_df_sub["is_test"] == 1].index, 5
        )
    if project_options.debug:
        test_index_chunks = [test_index_chunks[0]]
    no_test_pts_test = all(
        [re.match("test_[0-9]+.h5",i) is None for i in os.listdir(test_dh.loc)]
        )
    if no_test_pts_test:
        for idx, chunk in enumerate(test_index_chunks):
            print(f"Chunk: {idx+1}")
            with h5py.File(os.path.join(test_dh.loc, f'test_{idx}.h5'),'w') as h5f:
                for k in REQ_KEYS:
                    print(k)
                    h5f.create_dataset(
                        k,
                        data=process_data_lkp[k](
                            h5_file[k][chunk]
                            )
                        )
