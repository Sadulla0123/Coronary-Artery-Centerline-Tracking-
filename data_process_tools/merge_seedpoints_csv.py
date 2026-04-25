import os
import pandas as pd

PATCH_ROOT = os.path.join("patch_data", "seeds_patch")

POS_GAP = 100
NEG_GAP = 19

NEG_RATIO = 2
DATASETS = range(7)

POS_DIR = os.path.join(PATCH_ROOT, "positive", f"gp_{POS_GAP}")
NEG_DIR = os.path.join(PATCH_ROOT, "negative", f"gp_{NEG_GAP}")

for save_num in DATASETS:

    print("\nProcessing d", save_num)

    pos_csv = os.path.join(POS_DIR, f"d{save_num}_patch_info.csv")
    neg_csv = os.path.join(NEG_DIR, f"d{save_num}_patch_info.csv")

    if not os.path.exists(pos_csv) or not os.path.exists(neg_csv):
        print("Missing CSV for d", save_num)
        continue

    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)

    print("Positive:", len(df_pos))
    print("Negative:", len(df_neg))

    # balance negatives
    max_neg = min(len(df_neg), len(df_pos) * NEG_RATIO)
    df_neg = df_neg.sample(n=max_neg, random_state=42)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42)

    train_df = df_all.sample(frac=0.8, random_state=42)
    val_df = df_all.drop(train_df.index)

    train_csv = os.path.join(PATCH_ROOT, f"train_save_d{save_num}_train.csv")
    val_csv = os.path.join(PATCH_ROOT, f"train_save_d{save_num}_val.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print("Saved split for d", save_num)

print("\nSeedpoints merge complete.")