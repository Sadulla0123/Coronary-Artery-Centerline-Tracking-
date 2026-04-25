import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
gap_size = 1
neg_ratio = 2          # negatives per positive
datasets = range(7)    # d0 → d6

base_dir = os.path.join("patch_data", "ostia_patch")

pos_root = os.path.join(base_dir, "postive", f"gp_{gap_size}")   # typo kept
neg_root = os.path.join(base_dir, "negative", f"gp_{gap_size}")

print("Base dir:", base_dir)

for save_num in datasets:
    print("\n==============================")
    print(f"Processing dataset d{save_num}")
    print("==============================")

    pos_csv = os.path.join(pos_root, f"d{save_num}_patch_info.csv")
    neg_csv = os.path.join(neg_root, f"d{save_num}_patch_info.csv")

    if not os.path.exists(pos_csv):
        print(f"[SKIP] Missing positive CSV: {pos_csv}")
        continue

    if not os.path.exists(neg_csv):
        print(f"[SKIP] Missing negative CSV: {neg_csv}")
        continue

    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)

    print("Positive samples:", len(df_pos))
    print("Negative samples:", len(df_neg))

    # ------------------------------
    # Balance data
    # ------------------------------
    max_neg = min(len(df_neg), len(df_pos) * neg_ratio)
    df_neg = df_neg.sample(n=max_neg, random_state=42)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Total after balance:", len(df_all))

    # ------------------------------
    # Train / Val split (80 / 20)
    # ------------------------------
    train_df = df_all.sample(frac=0.8, random_state=42)
    val_df = df_all.drop(train_df.index)

    train_csv = os.path.join(base_dir, f"train_save_d{save_num}_train.csv")
    val_csv   = os.path.join(base_dir, f"train_save_d{save_num}_val.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print("Saved:")
    print(" ", train_csv)
    print(" ", val_csv)

print("\n✅ ALL DATASETS DONE (d0–d6)")
