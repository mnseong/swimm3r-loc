# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------
import numpy as np
import torch
from mast3r.retrieval.graph import make_pairs_fps


def make_pairs(
    imgs, scene_graph="complete", prefilter=None, symmetrize=True, sim_mat=None
):
    pairs = []
    if scene_graph == "complete":  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("swin") and not scene_graph.startswith("swinret"):
        iscyclic = not scene_graph.endswith("noncyclic")
        try:
            winsize = int(scene_graph.split("-")[1])
        except Exception as e:
            winsize = 3
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize + 1):
                idx = i + j
                if iscyclic:
                    idx = idx % len(imgs)  # explicit loop closure
                if idx >= len(imgs):
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("logwin"):
        iscyclic = not scene_graph.endswith("noncyclic")
        try:
            winsize = int(scene_graph.split("-")[1])
        except Exception as e:
            winsize = 3
        offsets = [2**i for i in range(winsize)]
        pairsid = set()
        for i in range(len(imgs)):
            ixs_l = [i - off for off in offsets]
            ixs_r = [i + off for off in offsets]
            for j in ixs_l + ixs_r:
                if iscyclic:
                    j = j % len(imgs)  # Explicit loop closure
                if j < 0 or j >= len(imgs) or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("oneref"):
        refid = int(scene_graph.split("-")[1]) if "-" in scene_graph else 0
        for j in range(len(imgs)):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    elif scene_graph.startswith("retrieval"):
        mode, Na, k = scene_graph.split("-")
        assert sim_mat is not None, "sim_mat is required for retrieval mode"

        fps_pairs, anchor_idxs = make_pairs_fps(
            sim_mat, Na=int(Na), tokK=int(k), dist_thresh=None
        )

        for i, j in fps_pairs:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("swinret"):
        assert sim_mat is not None, "sim_mat is required for swinret mode"
        for i, j in swinret_pair_ids(len(imgs), scene_graph, sim_mat):
            pairs.append((imgs[i], imgs[j]))
    else:
        raise ValueError(f"unrecognized value for {scene_graph=}")

    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith("seq"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith("cyc"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def swinret_pair_ids(n, scene_graph, sim_mat, seed=0):
    """Index pairs (i < j) of 'swinret-W-Na-k[-noncyclic]': the sliding window of W images in the order of the
    list (every image with the next W, around the end too unless noncyclic) united with the retrieval graph of
    make_pairs_fps (Na key images by farthest point sampling on 1 - sim_mat, all pairs between them, every other
    image with its nearest key image, and the k nearest images of every image). The window keeps a sequence
    connected where retrieval finds no partner, retrieval adds the pairs that see the same place across a gap in
    the list (held out frames) or a revisit. The first key image comes from a generator seeded with seed, the
    global numpy state is left as it was."""
    parts = scene_graph.split("-")
    if len(parts) not in (4, 5) or (len(parts) == 5 and parts[4] != "noncyclic"):
        raise ValueError(f"swinret needs 'swinret-W-Na-k[-noncyclic]', got {scene_graph!r}")
    win, na, k = int(parts[1]), int(parts[2]), int(parts[3])
    cyclic = len(parts) == 4
    sim = np.asarray(sim_mat)
    if sim.shape != (n, n):
        raise ValueError(f"sim_mat of shape {sim.shape} for {n} images")
    ids = set()
    for i in range(n):
        for d in range(1, win + 1):
            j = i + d
            if cyclic:
                j %= n
            if j >= n or j == i:
                continue
            ids.add((min(i, j), max(i, j)))
    n_swin = len(ids)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        ret, _ = make_pairs_fps(sim, Na=min(na, n), tokK=k, dist_thresh=None)
    finally:
        np.random.set_state(state)
    ret = {(int(min(i, j)), int(max(i, j))) for i, j in ret if i != j}
    ids |= ret
    far = sum(1 for i, j in ids if (min(j - i, n - (j - i)) if cyclic else j - i) > win)
    print(f"[swinret] {n} images: window {win} gives {n_swin} pairs, retrieval (Na {na}, k {k}) {len(ret)}, "
          f"union {len(ids)}, {len(ids) - n_swin} added by retrieval, {far} beyond the window")
    return sorted(ids)


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1["idx"], img2["idx"]) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(
        f">> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges"
    )
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)
