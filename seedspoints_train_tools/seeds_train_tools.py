import os
import sys
sys.path.append('..')

import torch
from torch.utils.data import ConcatDataset

from models.seedspoints_net import SeedspointsNet
from seeds_net_data_provider_aug import DataGenerater
from seeds_trainner import Trainer


# ==========================================================
# LOAD ALL DATASETS d0–d6
# ==========================================================
def get_dataset_all():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🔴 MUST END AT patch_data
    pre_fix_path = os.path.join(
        BASE_DIR,
        "..",
        "data_process_tools",
        "patch_data"
    )

    train_sets = []
    val_sets = []

    for save_num in range(7):

        train_csv = os.path.join(
            BASE_DIR,
            "..",
            "data_process_tools",
            "patch_data",
            "seeds_patch",
            f"train_save_d{save_num}_train.csv"
        )

        val_csv = os.path.join(
            BASE_DIR,
            "..",
            "data_process_tools",
            "patch_data",
            "seeds_patch",
            f"train_save_d{save_num}_val.csv"
        )

        if not os.path.exists(train_csv):
            print(f"[SKIP] Missing train CSV d{save_num}")
            continue

        if not os.path.exists(val_csv):
            print(f"[SKIP] Missing val CSV d{save_num}")
            continue

        print(f"Loading seedpoints dataset d{save_num}")

        train_sets.append(
            DataGenerater(train_csv, pre_fix_path, None, "train", None)
        )

        val_sets.append(
            DataGenerater(val_csv, pre_fix_path, None, "val", None)
        )

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)

    return train_dataset, val_dataset


# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':

    print("=== SEEDPOINTS TRAINING STARTED (d0–d6) ===")

    train_dataset, val_dataset = get_dataset_all()

    print("Total train dataset size:", len(train_dataset))
    print("Total val dataset size:", len(val_dataset))

    if len(train_dataset) == 0:
        raise RuntimeError("Dataset EMPTY — check merged CSVs")

    curr_model_name = "seedspoints_net_all"
    model = SeedspointsNet()

    batch_size = 64
    num_workers = 0   # safer on Ubuntu lab machines

    criterion = torch.nn.MSELoss()
    initial_lr = 0.001

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=0.001
    )

    trainer = Trainer(
        batch_size,
        num_workers,
        train_dataset,
        val_dataset,
        model,
        curr_model_name,
        optimizer,
        criterion,
        start_epoch=0,
        max_epoch=100,
        save_num=0,
        initial_lr=initial_lr
    )

    trainer.run_train()