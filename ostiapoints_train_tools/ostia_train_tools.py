# -*- coding: UTF-8 -*-

import os
import sys
sys.path.append('..')

import torch
from torch.utils.data import ConcatDataset

from models.ostiapoints_net import OstiapointsNet
from ostia_net_data_provider_aug import DataGenerater
from ostia_trainner import Trainer


def get_dataset_all():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    pre_fix_path = os.path.join(
        BASE_DIR, "..", "data_process_tools", "patch_data"
    )

    train_sets = []
    val_sets = []

    for save_num in range(7):  # d0–d6
        train_csv = os.path.join(
            BASE_DIR, "..", "data_process_tools", "patch_data",
            "ostia_patch", f"train_save_d{save_num}_train.csv"
        )

        val_csv = os.path.join(
            BASE_DIR, "..", "data_process_tools", "patch_data",
            "ostia_patch", f"train_save_d{save_num}_val.csv"
        )

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            print(f"[SKIP] Missing CSV for d{save_num}")
            continue

        print(f"Loading dataset d{save_num}")

        train_sets.append(
            DataGenerater(train_csv, pre_fix_path, None, "train", None)
        )

        val_sets.append(
            DataGenerater(val_csv, pre_fix_path, None, "val", None)
        )

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)

    return train_dataset, val_dataset


if __name__ == "__main__":

    print("=== OSTIA TRAINING (NO CSV MERGE) ===")

    train_dataset, val_dataset = get_dataset_all()

    print("Total train size:", len(train_dataset))
    print("Total val size:", len(val_dataset))

    model = OstiapointsNet()

    batch_size = 64
    num_workers = 0
    criterion = torch.nn.MSELoss()
    initial_lr = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.001
    )

    trainer = Trainer(
        batch_size,
        num_workers,
        train_dataset,
        val_dataset,
        model,
        "ostiapoints_net_all",
        optimizer,
        criterion,
        save_num=0,
        start_epoch=0,
        max_epoch=100,
        initial_lr=initial_lr
    )

    trainer.run_train()
