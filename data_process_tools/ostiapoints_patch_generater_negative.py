import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

np.random.seed(4)

from utils import resample, get_shell, get_proximity, get_max_boundr


# ==========================================================
# Create Data for One Dataset (positive or negative patches)
# ==========================================================
def creat_data(path_name, spacing_path, gap_size, save_num):
    # Load spacing info
    spacing_info = np.loadtxt(spacing_path, delimiter=",", dtype=np.float32)

    proximity_list = []
    patch_name = []

    i = save_num
    print(f"\n============================")
    print(f"Processing dataset {i}")
    print(f"============================")

    # ----------------------------------------------------------
    # Read CT Image
    # ----------------------------------------------------------
    image_folder = f"{path_name}0{i}"
    image_file = os.path.join(image_folder, f"image0{i}.nii.gz")

    if not os.path.exists(image_file):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_file}")

    src_array = sitk.GetArrayFromImage(
        sitk.ReadImage(image_file, sitk.sitkFloat32)
    )

    # read spacing
    spacing_x = spacing_info[i][0]
    spacing_y = spacing_info[i][1]
    spacing_z = spacing_info[i][2]

    # Resample to 1×1×1 spacing
    re_spacing_img, curr_spacing, resize_factor = resample(
        src_array,
        np.array([spacing_z, spacing_x, spacing_y]),
        np.array([1, 1, 1])
    )

    max_z, max_y, max_x = re_spacing_img.shape
    print("New resampled shape:", re_spacing_img.shape)

    # ----------------------------------------------------------
    # Read ostia points
    # ----------------------------------------------------------
    ostia_points = []

    for j in range(4):
        reference_path = os.path.join(
            f"{path_name}0{i}",
            f"vessel{j}",
            "pointS.txt"
        )

        if not os.path.exists(reference_path):
            print(f"[WARNING] Missing ostia point file: {reference_path}")
            continue

        txt_data = np.loadtxt(reference_path, dtype=np.float32)

        if j == 0 or j == 1:
            ostia_points.append(txt_data)
        else:
            ostia_points[1] = ostia_points[1] + txt_data

    # Average vessels 1,2,3 into one LAD ostium
    ostia_points[1] = ostia_points[1] / 3

    print("Final ostia points:", ostia_points)

    # ----------------------------------------------------------
    # Begin generating patches
    # ----------------------------------------------------------
    min_range = 17
    max_points = 100

    counter = 0
    record_set = set()

    for op in ostia_points:
        # compute max range allowed inside image
        max_range = get_max_boundr([max_x, max_y, max_z], op)

        for k in range(min_range, int(max_range + 1)):
            x_list, y_list, z_list = get_shell(max_points, k)

            record_set.add((int(round(op[0])), int(round(op[1])), int(round(op[2]))))

            for m in range(len(x_list)):
                new_x = int(round(op[0] + x_list[m]))
                new_y = int(round(op[1] + y_list[m]))
                new_z = int(round(op[2] + z_list[m]))

                check_temp = (new_x, new_y, new_z)

                if check_temp in record_set:
                    continue
                record_set.add(check_temp)

                target_point = np.array([new_x, new_y, new_z])
                min_dis = np.linalg.norm(target_point - op)

                curr_proximity = get_proximity(min_dis, cutoff_value=16)

                cut_size = 9
                left_x = new_x - cut_size
                right_x = new_x + cut_size
                left_y = new_y - cut_size
                right_y = new_y + cut_size
                left_z = new_z - cut_size
                right_z = new_z + cut_size

                # boundary check
                if (left_x < 0 or right_x >= max_x or
                    left_y < 0 or right_y >= max_y or
                    left_z < 0 or right_z >= max_z or
                    curr_proximity > 0):
                    continue

                # Extract patch cube
                new_src_arr = np.zeros((cut_size*2+1, cut_size*2+1, cut_size*2+1))

                for ind in range(left_z, right_z + 1):
                    src_temp = re_spacing_img[ind].copy()
                    new_src_arr[ind - left_z] = src_temp[left_y:right_y+1, left_x:right_x+1]

                # Save patch
                save_folder = os.path.join(
                    "patch_data", "ostia_patch", "negative",
                    f"gp_{gap_size}", f"d{i}"
                )
                os.makedirs(save_folder, exist_ok=True)

                file_name = f"d_{i}_x_{new_x}_y_{new_y}_z_{new_z}.nii.gz"
                full_patch_path = os.path.join(save_folder, file_name)

                sitk.WriteImage(sitk.GetImageFromArray(new_src_arr), full_patch_path)

                # Append metadata
                patch_name.append(
                    os.path.join(
                        "ostia_patch", "negative",
                        f"gp_{gap_size}", f"d{i}", file_name
                    )
                )
                proximity_list.append(curr_proximity)
                counter += 1

    return patch_name, proximity_list


# ==========================================================
# Create patch CSV for all datasets
# ==========================================================
def create_patch_images(path_name, spacing_path, gap_size):
    for i in range(8):
        patch_name, proximity_list = creat_data(path_name, spacing_path, gap_size, i)

        df = pd.DataFrame({
            "patch_name": patch_name,
            "proximity": proximity_list
        })

        csv_folder = os.path.join("patch_data", "ostia_patch", "negative", f"gp_{gap_size}")
        os.makedirs(csv_folder, exist_ok=True)

        csv_path = os.path.join(csv_folder, f"d{i}_patch_info.csv")
        df.to_csv(csv_path, index=False)

        print(f"[OK] Saved patch CSV → {csv_path}")


# ==========================================================
# Run Full Pipeline
# ==========================================================
path_name = 'train_data/dataset'
spacing_path = 'spacing_info.csv'
gap_size = 1

create_patch_images(path_name, spacing_path, gap_size)
