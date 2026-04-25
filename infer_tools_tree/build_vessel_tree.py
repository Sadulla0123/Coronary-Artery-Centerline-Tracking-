import numpy as np
import copy
from setting import infer_model, device, spacing, re_spacing_img, resize_factor, max_points
from utils import get_spacing_res2, data_preprocess, prob_terminates, get_shell, get_angle
from tree import TreeNode
import torch


# ----------------------------------------------------------
# Check if point is near ostia
# ----------------------------------------------------------
def near_ostia(point, ostias, thr=10):
    for o in ostias:
        if np.linalg.norm(np.array(point) - np.array(o)) < thr:
            return True
    return False


# ----------------------------------------------------------
# BFS search
# ----------------------------------------------------------
def search_tree(root: TreeNode, point):

    queue = [root]

    while queue:

        node = queue.pop(0)

        if node.value is not None:

            dis = np.linalg.norm(node.value - point, axis=1)

            if np.min(dis) < 3:
                return node

        for c in node.child_list:
            queue.append(c)

    return None


# ----------------------------------------------------------
# Inference step
# ----------------------------------------------------------
def infer(start):

    cut = 9

    max_z, max_x, max_y = re_spacing_img.shape

    cx = get_spacing_res2(start[0], spacing[0], resize_factor[1])
    cy = get_spacing_res2(start[1], spacing[1], resize_factor[2])
    cz = get_spacing_res2(start[2], spacing[2], resize_factor[0])

    lx = cx - cut
    rx = cx + cut
    ly = cy - cut
    ry = cy + cut
    lz = cz - cut
    rz = cz + cut

    if lx < 0 or ly < 0 or lz < 0:
        return None

    if rx >= max_x or ry >= max_y or rz >= max_z:
        return None

    patch = np.zeros((19,19,19))

    for z in range(lz,rz+1):
        patch[z-lz] = re_spacing_img[z][ly:ry+1, lx:rx+1]

    input_data = data_preprocess(patch)

    outputs = infer_model(input_data.to(device).float())

    outputs = outputs.view((1,max_points+1))

    outputs_dir = outputs[:,:max_points]
    outputs_r = outputs[:,-1]

    outputs_dir = torch.nn.functional.softmax(outputs_dir,1)

    idx = np.argsort(outputs_dir.cpu().detach().numpy()[0])[::-1]

    prob = prob_terminates(outputs_dir,max_points).cpu().detach().numpy()[0]

    r = outputs_r.cpu().detach().numpy()[0]

    sx,sy,sz = get_shell(max_points,r)

    return [sx,sy,sz], idx, r, prob


# ----------------------------------------------------------
# Move step
# ----------------------------------------------------------
def move(start,shell,idx,move_dir):

    sx,sy,sz = shell

    for i in idx:

        ang = get_angle(
            np.array([sx[i],sy[i],sz[i]]),
            np.array(move_dir)
        )

        if ang <= 60:

            nx = start[0] + sx[i]
            ny = start[1] + sy[i]
            nz = start[2] + sz[i]

            return [sx[i],sy[i],sz[i]],[nx,ny,nz]

    return move_dir,start


# ----------------------------------------------------------
# First node
# ----------------------------------------------------------
def search_first_node(start,prob_records):

    shell,idx,r,prob = infer(start)

    prob_records.pop(0)
    prob_records.append(prob)

    sx,sy,sz = shell

    f = [sx[idx[0]],sy[idx[0]],sz[idx[0]]]

    for i in idx[1:]:

        ang = get_angle(
            np.array([sx[i],sy[i],sz[i]]),
            np.array(f)
        )

        if ang >= 90:

            b = [sx[i],sy[i],sz[i]]
            break

    direction = {}

    direction["forward"] = [
        start[0]+f[0],
        start[1]+f[1],
        start[2]+f[2]
    ]

    direction["forward_vector"] = f

    direction["backward"] = [
        start[0]+b[0],
        start[1]+b[1],
        start[2]+b[2]
    ]

    direction["backward_vector"] = b

    return direction,prob_records,r


# ----------------------------------------------------------
# Direction tracking
# ----------------------------------------------------------
def search_one_direction(start,move_dir,prob_records,points,radius):

    step = 0

    while True:

        prob_mean = sum(prob_records)/len(prob_records)

        if step>10 and prob_mean>0.85:
            break

        result = infer(start)

        if result is None:
            break

        shell,idx,r,prob = result

        points.append(start)
        radius.append(r)

        prob_records.pop(0)
        prob_records.append(prob)

        move_dir,start = move(start,shell,idx,move_dir)

        step +=1

        if step>300:
            break

    return points,radius


# ----------------------------------------------------------
# Interpolation
# ----------------------------------------------------------
def interpolation(points,r):

    if len(points)<2:
        return np.array(points)

    res = []

    for i in range(len(points)-1):

        p1 = points[i]
        p2 = points[i+1]

        num = max(2,int(r[i]/0.03))

        tmp = np.linspace(p1,p2,num=num)

        res.append(tmp)

    return np.vstack(res)


# ----------------------------------------------------------
# Search vessel
# ----------------------------------------------------------
def search_line(seed,r,direction,prob_records,root):

    p=[seed]
    rad=[r]

    pf,rf = search_one_direction(
        direction["forward"],
        direction["forward_vector"],
        copy.deepcopy(prob_records),
        copy.deepcopy(p),
        copy.deepcopy(rad)
    )

    pb,rb = search_one_direction(
        direction["backward"],
        direction["backward_vector"],
        copy.deepcopy(prob_records),
        copy.deepcopy(p),
        copy.deepcopy(rad)
    )

    points = pf[::-1] + pb
    r_all = rf[::-1] + rb

    if len(points)<5:
        return False

    end = points[-1]

    if not near_ostia(end,root.value):
        return False

    res = interpolation(points,r_all)

    root.add_child(TreeNode(res,start_point_index=None))

    return True


# ----------------------------------------------------------
# Vessel tree builder
# ----------------------------------------------------------
def build_vessel_tree(seeds,root):

    prob_records=[0,0,0]

    for seed in seeds:

        if search_tree(root,seed) is not None:
            continue

        direction,prob_records,r = search_first_node(seed,prob_records)

        search_line(seed,r,direction,prob_records,root)


# ----------------------------------------------------------
# DFS
# ----------------------------------------------------------
def dfs_search_tree(root):

    vessels=[]

    stack=[root]

    while stack:

        node = stack.pop()

        for c in node.child_list:

            vessels.append(c)

            stack.append(c)

    return vessels