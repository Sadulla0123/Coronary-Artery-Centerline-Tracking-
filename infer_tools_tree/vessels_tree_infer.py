import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from generate_seeds_ositas import search_seeds_ostias
from setting import setting_info
from build_vessel_tree import build_vessel_tree, TreeNode, dfs_search_tree
from utils import save_info


# -------------------------------------------------
# Create output folders
# -------------------------------------------------
os.makedirs(setting_info["seeds_gen_info_to_save"], exist_ok=True)
os.makedirs(setting_info["ostias_gen_info_to_save"], exist_ok=True)
os.makedirs(setting_info["infer_line_to_save"], exist_ok=True)
os.makedirs(setting_info["fig_to_save"], exist_ok=True)


# -------------------------------------------------
# Detect seeds and ostias
# -------------------------------------------------
print("search seeds and ostias")

res_seeds, res_ostia = search_seeds_ostias()

seeds_csv = os.path.join(setting_info["seeds_gen_info_to_save"], "seeds.csv")
ostias_csv = os.path.join(setting_info["ostias_gen_info_to_save"], "ostias.csv")

save_info(res_seeds, seeds_csv)
save_info(res_ostia, ostias_csv)

seeds = pd.read_csv(seeds_csv)[["x", "y", "z"]].values
ostia_candidates = pd.read_csv(ostias_csv)[["x", "y", "z"]].values


# -------------------------------------------------
# Cluster seeds (reduce noisy seeds)
# -------------------------------------------------
print("clustering seeds")

clustered_seeds = []
min_dist = 6

for s in seeds:
    keep = True
    for cs in clustered_seeds:
        if np.linalg.norm(s - cs) < min_dist:
            keep = False
            break
    if keep:
        clustered_seeds.append(s)

seeds = np.array(clustered_seeds)

print("Seeds after clustering:", len(seeds))


# -------------------------------------------------
# Select 2 farthest ostia
# -------------------------------------------------
print("selecting ostia")

if len(ostia_candidates) < 2:
    print("Not enough ostia detected")
    exit()

ostias = []

first = ostia_candidates[0]
ostias.append(first.tolist())

for node in ostia_candidates:
    if np.linalg.norm(node - first) > 10:
        ostias.append(node.tolist())
        break

if len(ostias) < 2:
    print("Could not find 2 separate ostia")
    exit()

print("Selected ostias:", ostias)


# -------------------------------------------------
# Build vessel tree
# -------------------------------------------------
print("build vessel tree")

root = TreeNode(np.array(ostias), start_point_index=None)

build_vessel_tree(seeds, root)

single_tree = dfs_search_tree(root)


# -------------------------------------------------
# Postprocess vessels
# -------------------------------------------------
vessel_tree_postprocess = []

for vessel in single_tree:

    if vessel is None:
        continue

    if len(vessel) < 10:
        continue

    vessel_tree_postprocess.append(vessel)


print("Total vessels:", len(vessel_tree_postprocess))


# -------------------------------------------------
# Save vessels
# -------------------------------------------------
save_path = setting_info["infer_line_to_save"]

for i, vessel in enumerate(vessel_tree_postprocess):
    np.savetxt(os.path.join(save_path, f"vessel_{i}.txt"), vessel)


# -------------------------------------------------
# Plot inferred vessels
# -------------------------------------------------
fig_dir = setting_info["fig_to_save"]

if len(vessel_tree_postprocess) > 0:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, vessel in enumerate(vessel_tree_postprocess):
        ax.plot(vessel[:,0], vessel[:,1], vessel[:,2], label=f"infer {i}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    plt.savefig(os.path.join(fig_dir,"infer_tree_new.jpg"))
    plt.close()

else:
    print("No vessels to plot")


# -------------------------------------------------
# Plot reference vs inference
# -------------------------------------------------
reference_path = setting_info["reference_path"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# reference vessels
for i in range(4):

    ref_file = os.path.join(reference_path, f"vessel{i}/reference.txt")

    if os.path.exists(ref_file):

        ref = np.loadtxt(ref_file)

        ax.plot(ref[:,0], ref[:,1], ref[:,2], label=f"ref {i}")


# inferred vessels
for i, vessel in enumerate(vessel_tree_postprocess):
    ax.plot(vessel[:,0], vessel[:,1], vessel[:,2], label=f"infer {i}")


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

plt.savefig(os.path.join(fig_dir,"refer_infer_tree.jpg"))
plt.close()


# -------------------------------------------------
# Plot seeds
# -------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(seeds[:,0], seeds[:,1], seeds[:,2], s=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig(os.path.join(fig_dir,"seeds_points.jpg"))
plt.close()


print("Inference finished.")