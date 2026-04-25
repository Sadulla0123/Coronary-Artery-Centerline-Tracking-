import os
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

MAX_POINTS = 500
GAP_SIZE = 1
DATASETS = range(7)   # d0 → d6

BASE_DIR = os.path.join(
    "patch_data",
    "centerline_patch",
    "no_offset",
    f"point_{MAX_POINTS}_gp_{GAP_SIZE}"
)

OUTPUT_DIR = os.path.join(
    "patch_data",
    "centerline_patch"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Input folder:", BASE_DIR)
print("Output folder:", OUTPUT_DIR)

# ==========================================================
# SPLIT LOOP
# ==========================================================

for save_num in DATASETS:

    print("\n==============================")
    print(f"Splitting dataset d{save_num}")
    print("==============================")

    input_csv = os.path.join(
        BASE_DIR,
        f"d{save_num}_patch_info_{MAX_POINTS}.csv"
    )

    if not os.path.exists(input_csv):
        print(f"[SKIP] CSV not found: {input_csv}")
        continue

    df = pd.read_csv(input_csv)

    print("Total samples:", len(df))

    # ------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ------------------------------------------------------
    # Train / Val split (80 / 20)
    # ------------------------------------------------------
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    train_csv = os.path.join(
        OUTPUT_DIR,
        f"train_save_d{save_num}_train.csv"
    )

    val_csv = os.path.join(
        OUTPUT_DIR,
        f"train_save_d{save_num}_val.csv"
    )

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print("Saved:")
    print(" ", train_csv)
    print(" ", val_csv)

print("\n✅ Centerline train/val split complete (d0–d6)")