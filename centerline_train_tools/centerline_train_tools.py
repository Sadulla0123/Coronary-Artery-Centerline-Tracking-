import os
import sys
sys.path.append('..')

import torch
from torch.utils.data import ConcatDataset

from models.centerline_net import CenterlineNet
from data_provider_argu import DataGenerater
from centerline_trainner import Trainer


# ==========================================================
# LOSS FUNCTION (same as original repo)
# ==========================================================
def cross_entropy(a, y):
    epsilon = 1e-9
    return torch.mean(
        torch.sum(
            -y * torch.log10(a + epsilon) -
            (1 - y) * torch.log10(1 - a + epsilon),
            dim=1
        )
    )


# ==========================================================
# LOAD ALL DATASETS d0–d6
# ==========================================================
def get_dataset_all():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
            "centerline_patch",
            f"train_save_d{save_num}_train.csv"
        )

        val_csv = os.path.join(
            BASE_DIR,
            "..",
            "data_process_tools",
            "patch_data",
            "centerline_patch",
            f"train_save_d{save_num}_val.csv"
        )

        if not os.path.exists(train_csv):
            print(f"[SKIP] Missing train CSV d{save_num}")
            continue

        if not os.path.exists(val_csv):
            print(f"[SKIP] Missing val CSV d{save_num}")
            continue

        print(f"Loading dataset d{save_num}")

        train_sets.append(
            DataGenerater(train_csv, pre_fix_path, 500, None, "train", None)
        )

        val_sets.append(
            DataGenerater(val_csv, pre_fix_path, 500, None, "val", None)
        )

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)

    return train_dataset, val_dataset


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    print("=== CENTERLINE TRAINING STARTED (d0–d6) ===")

    train_dataset, val_dataset = get_dataset_all()

    print("Total train dataset size:", len(train_dataset))
    print("Total val dataset size:", len(val_dataset))

    if len(train_dataset) == 0:
        raise RuntimeError("Dataset EMPTY — check split CSVs")

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------
    max_points = 500
    model = CenterlineNet(n_classes=max_points)

    batch_size = 64
    num_workers = 0
    initial_lr = 0.001
    criterion = cross_entropy

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=0.001
    )

    print("=== INITIALIZING TRAINER ===")

    trainer = Trainer(
        batch_size,
        num_workers,
        train_dataset,
        val_dataset,
        model,
        "centerline_net_all",
        optimizer,
        criterion,
        max_points,
        save_num=0,
        start_epoch=0,
        max_epoch=100,
        initial_lr=initial_lr
    )

    print("=== STARTING TRAINING ===")
    trainer.run_train()
    print("=== TRAINING FINISHED ===")