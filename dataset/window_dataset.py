"""Multi frame windows for the Gaussian map branch.

One sample is an anchor frame, a random number of further context frames that overlap with it
and a few held out target frames between them. Frames keep their aspect ratio and are resized
like at inference (long edge 512, centre crop to a multiple of 16), because the Gaussians are
placed with the geometry of the network and a stretched image has no consistent geometry.

With probability jerlov_p the underwater frames of the whole window are replaced by frames
rendered from the restored images and the metric depth with one random medium, the same
synthesis the pair branch of the training uses.

With frame_times a sample also carries the capture time of every frame in seconds. Sea-Thru
sites are photo sets without a time, their windows are untimed and carry zeros.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

DS_TAG = "downsampled_016"
IMG_DIR = "images_jpg_undistorted"
FLSEA_FPS = 10.0


def frame_times(names, fps):
    """Seconds from the first frame of the scene, float64 (n,). names are the sorted '<int>.jpg' of a scene.

    Stems of at least 1e15 are capture time stamps with ten integer digits of unix seconds,
    smaller stems are frame indices at fps. The difference is taken in integers, so long time
    stamps lose no precision.
    """
    stems = [int(n[:-4]) for n in names]
    if not stems:
        return np.zeros(0, dtype=np.float64)
    s0 = min(stems)
    unit = 10.0 ** (len(str(s0)) - 10) if s0 >= 10 ** 15 else float(fps)
    return np.array([s - s0 for s in stems], dtype=np.float64) / unit


def resize_crop_box(w0, h0, long_edge=512, patch=16):
    """Size after the resize and the crop box, the rule of dust3r.utils.image.load_images."""
    s = long_edge / max(w0, h0)
    wn, hn = int(round(w0 * s)), int(round(h0 * s))
    cx, cy = wn // 2, hn // 2
    halfw, halfh = ((2 * cx) // patch) * (patch // 2), ((2 * cy) // patch) * (patch // 2)
    return (wn, hn), (cx - halfw, cy - halfh, cx + halfw, cy + halfh)


def load_rgb(path, long_edge):
    img = Image.open(path).convert("RGB")
    size, box = resize_crop_box(*img.size, long_edge=long_edge)
    interp = Image.LANCZOS if max(img.size) > long_edge else Image.BICUBIC
    return np.asarray(img.resize(size, interp).crop(box), dtype=np.uint8)


def load_depth(path, long_edge):
    z = np.load(path).astype(np.float32)
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0).astype(np.float32)
    h0, w0 = z.shape
    size, box = resize_crop_box(w0, h0, long_edge=long_edge)
    return np.asarray(Image.fromarray(z).resize(size, Image.NEAREST).crop(box), dtype=np.float32)


class UnderwaterWindowDataset(IterableDataset):
    def __init__(
        self,
        root_path,
        dataset_config_yml,
        precomputed_pairs=None,
        long_edge=512,
        ctx_min=4,
        ctx_max=12,
        n_targets=3,
        radii=(6, 12, 24, 48),
        jerlov_p=0.5,
        jerlov_kwargs=None,
        jerlov_noise=0.01,
        min_transmission=0.25,
        seed=0,
        frame_times=False,
        fps=FLSEA_FPS,
    ):
        super().__init__()
        self.root = Path(root_path)
        self.long_edge = int(long_edge)
        self.ctx_min, self.ctx_max = int(ctx_min), int(ctx_max)
        assert 2 <= self.ctx_min <= self.ctx_max
        self.n_targets = int(n_targets)
        self.radii = [int(r) for r in radii]
        self.jerlov_p = float(jerlov_p)
        self.jerlov_kwargs = dict(jerlov_kwargs or {})
        self.jerlov_noise = float(jerlov_noise)
        self.min_transmission = float(min_transmission)
        self.seed = int(seed)
        self.frame_times = bool(frame_times)
        self.fps = float(fps)

        with open(dataset_config_yml, "r") as f:
            cfg = yaml.safe_load(f)
        self.scenes = []
        for scene, c in cfg.items():
            noisy, clean = c["noisy_iso_et"][0], c["clean_iso_et"]
            ndir = self.root / scene / noisy / DS_TAG / IMG_DIR
            cdir = self.root / scene / clean / DS_TAG / IMG_DIR
            ddir = self.root / scene / "depth"
            if not ndir.is_dir():
                print(f"[window] missing dir: {ndir}")
                continue
            names = sorted(n for n in os.listdir(ndir) if n.endswith(".jpg"))
            stem_to_pos = {int(n[:-4]): i for i, n in enumerate(names)}
            pairs = None
            if precomputed_pairs:
                pp = self.root / scene / clean / DS_TAG / precomputed_pairs
                if pp.is_file():
                    raw = json.load(open(pp, "r"))
                    pairs = {}
                    for k, v in raw.items():
                        if int(k) in stem_to_pos:
                            pairs[stem_to_pos[int(k)]] = np.array(
                                sorted({stem_to_pos[int(j)] for j in v if int(j) in stem_to_pos}), dtype=np.int64
                            )
            self.scenes.append(dict(
                name=scene, names=names, ndir=ndir, cdir=cdir, ddir=ddir, pairs=pairs,
                video=scene.startswith("flsea_"),
            ))
        if not self.scenes:
            raise RuntimeError("no scene found for the window dataset")
        n = np.array([len(s["names"]) for s in self.scenes], dtype=np.float64)
        self.scene_p = n / n.sum()
        print(f"[window] {len(self.scenes)} scenes, {int(n.sum())} frames, contexts {self.ctx_min}-{self.ctx_max}, "
              f"targets {self.n_targets}, radii {self.radii}, jerlov_p {self.jerlov_p}")
        if self.frame_times:
            for sc in self.scenes:
                sc["times"] = self.scene_times(sc)
            timed = [sc for sc in self.scenes if sc["video"]]
            step = [float(np.median(np.diff(sc["times"]))) for sc in timed if len(sc["times"]) > 1]
            print(f"[window] frame times at {self.fps} fps, {len(timed)} timed scenes, "
                  f"median time step {min(step, default=0.0):.4f}-{max(step, default=0.0):.4f} s")

    def scene_times(self, sc):
        """Seconds of every frame of a scene, zeros for a photo set."""
        if sc["video"]:
            return frame_times(sc["names"], self.fps)
        return np.zeros(len(sc["names"]), dtype=np.float64)

    def candidates(self, sc, a, rng):
        n = len(sc["names"])
        if sc["video"]:
            r = int(rng.choice(self.radii))
            lo, hi = max(0, a - r), min(n - 1, a + r)
            cand = np.arange(lo, hi + 1)
            if sc["pairs"] is not None and a in sc["pairs"]:
                cand = np.intersect1d(cand, sc["pairs"][a])
        elif sc["pairs"] is not None and a in sc["pairs"]:
            cand = sc["pairs"][a]
        else:
            r = max(2, min(self.radii))
            cand = np.arange(max(0, a - r), min(n - 1, a + r) + 1)
        return cand[cand != a]

    def sample_indices(self, rng):
        for _ in range(50):
            sc = self.scenes[int(rng.choice(len(self.scenes), p=self.scene_p))]
            a = int(rng.integers(len(sc["names"])))
            cand = self.candidates(sc, a, rng)
            if len(cand) < 2:
                continue
            n_tgt = int(min(self.n_targets, max(1, len(cand) // 3)))
            n_ctx = int(min(rng.integers(self.ctx_min - 1, self.ctx_max), len(cand) - n_tgt))
            chosen = np.sort(rng.choice(cand, size=n_ctx + n_tgt, replace=False))
            inner = chosen[1:-1] if len(chosen) >= n_tgt + 2 else chosen
            tgt = np.sort(rng.choice(inner, size=n_tgt, replace=False))
            ctx = np.array([c for c in chosen if c not in set(tgt.tolist())], dtype=np.int64)
            if len(ctx) < 1:
                continue
            return sc, [a] + ctx.tolist() + tgt.tolist(), 1 + len(ctx)
        raise RuntimeError("could not sample a window")

    def load(self, sc, frames, n_ctx, rng):
        raw = np.stack([load_rgb(sc["ndir"] / sc["names"][i], self.long_edge) for i in frames])
        clean = np.stack([load_rgb(sc["cdir"] / sc["names"][i], self.long_edge) for i in frames])
        raw = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
        clean = torch.from_numpy(clean).permute(0, 3, 1, 2).contiguous()
        synth = False
        if self.jerlov_p > 0 and rng.random() < self.jerlov_p:
            from losses.physics_regression import jerlov_render, sample_jerlov_medium
            depth = np.stack([
                load_depth(sc["ddir"] / (sc["names"][i][:-4] + ".npy"), self.long_edge) for i in frames
            ])
            depth = torch.from_numpy(depth)
            if depth.shape[-2:] == clean.shape[-2:] and float((depth > 0).float().mean()) > 0.5:
                # Pixels without depth are mostly open water, they get the far range of the frame.
                far = torch.quantile(depth[depth > 0][::17], 0.99).clamp(max=50.0)
                depth = torch.where(depth > 0, depth, far)
                # A medium that leaves less than a quarter of the direct signal in every channel
                # at the median range gives frames without content, such a draw is repeated.
                zmed = depth[::, ::7, ::7].median()
                for _ in range(8):
                    bd, bb, binf = sample_jerlov_medium(1, device="cpu", **self.jerlov_kwargs)
                    if float(torch.exp(-bd[0] * zmed).max()) >= self.min_transmission:
                        f = len(frames)
                        j = clean.float() / 127.5 - 1.0
                        syn = jerlov_render(j, depth, bd.expand(f, 3), bb.expand(f, 3), binf.expand(f, 3),
                                            noise_std=self.jerlov_noise)
                        raw = ((syn + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
                        synth = True
                        break
        out = dict(raw=raw, clean=clean, n_ctx=int(n_ctx), scene=sc["name"],
                   frames=torch.tensor(frames, dtype=torch.int64), synth=bool(synth))
        if self.frame_times:
            out["times"] = torch.from_numpy(sc["times"][np.asarray(frames, dtype=np.int64)])
            out["timed"] = bool(sc["video"])
        return out

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, wid])
        torch.manual_seed(self.seed * 1000003 + wid)
        while True:
            try:
                sc, frames, n_ctx = self.sample_indices(rng)
                yield self.load(sc, frames, n_ctx, rng)
            except (OSError, ValueError) as e:
                print(f"[window] skipped a window: {type(e).__name__}: {e}")
                continue
