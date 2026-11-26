"""Medium field of a feed forward map: the water of a scene as a function of the camera, fitted offline next to
the Gaussians, so that the underwater render of any pose needs no map frame to borrow the medium from.

The physics head gives every map frame its own medium: beta_D(z) = a exp(b z) + c exp(d z) per channel
(params_D, 4 x 3), beta_B (params_B), and the veiling light as a map B_inf(x, y) over the image (a quadratic in the
image coordinates, so it depends on the viewing direction). The field keeps
  - B_inf as a small MLP of the world direction of the ray and the camera centre, both with low frequency Fourier
    features, sigmoid output; it is fitted to the B_inf maps of the map frames sampled on a grid of rays,
  - beta_D and beta_B as one medium of the scene, the mean of the per frame parameters (attenuation is a property
    of the water; the spread of the per frame predictions is reported as a diagnostic).
medium_at(c2w, focal, pp, h, w) returns the medium of a camera in the format of training/physics_splat.py
(params_D, params_B, B_inf, B_inf_map), so the renderer is unchanged.

ViewMedium is the feed forward variant with no fitting: the B_inf of a ray is read from the B_inf maps of the map
frames that saw its direction, weighted by angle to their optical axis and distance between the cameras.
"""

import math

import torch
import torch.nn as nn


def fourier(x, n_freq):
    """x (..., k) -> (..., k (1 + 2 n_freq)), frequencies 1, 2, 4, ... times pi."""
    out = [x]
    for i in range(n_freq):
        f = (2.0 ** i) * math.pi
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, dim=-1)


def pixel_dirs(c2w, focal, pp, h, w, step=1):
    """World unit directions (h', w', 3) of the pixel centres (j + 0.5) every step pixels, OpenCV camera."""
    dev, dt = c2w.device, c2w.dtype
    v, u = torch.meshgrid(torch.arange(0, h, step, device=dev, dtype=dt) + 0.5,
                          torch.arange(0, w, step, device=dev, dtype=dt) + 0.5, indexing="ij")
    d = torch.stack([(u - pp[0]) / focal, (v - pp[1]) / focal, torch.ones_like(u)], dim=-1)
    d = d @ c2w[:3, :3].T
    return d / d.norm(dim=-1, keepdim=True)


class MediumField(nn.Module):
    def __init__(self, width=64, depth=3, dir_freq=2, pos_freq=1):
        super().__init__()
        self.cfg = dict(width=width, depth=depth, dir_freq=dir_freq, pos_freq=pos_freq)
        in_dim = 3 * (1 + 2 * dir_freq) + 3 * (1 + 2 * pos_freq)
        layers, d = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 3))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("centre", torch.zeros(3))
        self.register_buffer("radius", torch.ones(()))
        self.register_buffer("params_D", torch.zeros(4, 3))
        self.register_buffer("params_B", torch.zeros(4, 3))
        self.register_buffer("binf_mean", torch.full((3,), 0.5))
        self.register_buffer("beta_D_scale", torch.ones(3))
        self.has_scale = False

    def forward(self, dirs, pos):
        """B_inf (..., 3) of rays with world directions dirs (..., 3) from camera centres pos (..., 3)."""
        p = (pos - self.centre) / self.radius
        x = torch.cat([fourier(dirs, self.cfg["dir_freq"]), fourier(p, self.cfg["pos_freq"])], dim=-1)
        logit = torch.log(self.binf_mean / (1.0 - self.binf_mean)).to(x.dtype)
        return torch.sigmoid(self.mlp(x) + logit)

    @torch.no_grad()
    def medium_at(self, c2w, focal, pp, h, w):
        """Medium of one camera: params_D, params_B (4, 3), B_inf (3,) the mean of the map, B_inf_map (3, h, w)."""
        dirs = pixel_dirs(c2w.to(self.centre.dtype), float(focal), pp, h, w)
        mp = self(dirs, c2w[:3, 3].to(self.centre.dtype).expand_as(dirs)).permute(2, 0, 1).contiguous()
        out = dict(params_D=self.params_D.clone(), params_B=self.params_B.clone(), B_inf=mp.mean((1, 2)), B_inf_map=mp)
        if self.has_scale:
            out["beta_D_scale"] = self.beta_D_scale.clone()
        return out

    def state(self):
        return dict(cfg=self.cfg, state=self.state_dict(), has_scale=self.has_scale)

    @classmethod
    def from_state(cls, st):
        m = cls(**st["cfg"])
        m.load_state_dict(st["state"])
        m.has_scale = bool(st["has_scale"])
        return m


class ViewMedium:
    """The medium of a camera from the per frame media of the map frames, with no fitting: a ray takes the B_inf
    of the map frames that saw its direction, every frame's B_inf map read at the pixel where the ray direction
    falls in that frame, weighted by exp(-theta^2 / 2 sigma_theta^2) exp(-r^2 / 2 (sigma_pos R)^2) with theta the
    angle between the ray and the frame's optical axis, r the distance between the camera centres and R the
    radius of the trajectory. Only frames whose image holds the direction (with a margin of margin times the
    image size) count; a direction no frame saw takes the angle weighted mean of the maps read at their border.
    beta_D and beta_B are the mean of the map frames as in the field. Same interface as MediumField.medium_at."""

    def __init__(self, med, c2w, intr, h, w, idx=None, sigma_deg=20.0, sigma_pos=0.5, margin=0.05):
        idx = list(range(c2w.shape[0])) if idx is None else [int(i) for i in idx]
        self.maps = (med["B_inf_map"][idx] if "B_inf_map" in med else med["B_inf"][idx][:, :, None, None].expand(len(idx), 3, h, w)).float()
        self.R = c2w[idx, :3, :3].float()
        self.t = c2w[idx, :3, 3].float()
        self.f = intr[idx, 0, 0].float()
        self.cx = intr[idx, 0, 2].float() + 0.5
        self.cy = intr[idx, 1, 2].float() + 0.5
        self.hw = (self.maps.shape[-2], self.maps.shape[-1])
        self.params_D = med["params_D"][idx].float().mean(0)
        self.params_B = med["params_B"][idx].float().mean(0)
        self.scale = torch.as_tensor(med["beta_D_scale"], dtype=torch.float32).reshape(3) if "beta_D_scale" in med else None
        self.radius = float((self.t - self.t.mean(0)).norm(dim=-1).max().clamp_min(1e-6))
        self.sigma = math.radians(sigma_deg)
        self.sigma_pos = sigma_pos
        self.margin = margin
        self.binf_mean = self.maps.mean((0, 2, 3))

    @torch.no_grad()
    def medium_at(self, c2w, focal, pp, h, w):
        import torch.nn.functional as F
        d = pixel_dirs(c2w.float(), float(focal), pp, h, w)
        p = c2w[:3, 3].float()
        H, W = self.hw
        acc_in = torch.zeros(3, h, w, device=d.device)
        w_in = torch.zeros(h, w, device=d.device)
        acc_all = torch.zeros(3, h, w, device=d.device)
        w_all = torch.zeros(h, w, device=d.device)
        for i in range(self.maps.shape[0]):
            dc = d @ self.R[i]
            z = dc[..., 2]
            front = z > 1e-6
            zs = torch.where(front, z, torch.ones_like(z))
            u = self.f[i] * dc[..., 0] / zs + self.cx[i]
            v = self.f[i] * dc[..., 1] / zs + self.cy[i]
            ang = torch.acos(dc[..., 2].clamp(-1.0, 1.0))
            wt = torch.exp(-0.5 * (ang / self.sigma) ** 2)
            wt = wt * math.exp(-0.5 * (float((p - self.t[i]).norm()) / (self.sigma_pos * self.radius)) ** 2)
            inside = front & (u > -self.margin * W) & (u < (1 + self.margin) * W) & (v > -self.margin * H) & (v < (1 + self.margin) * H)
            grid = torch.stack([2.0 * u / W - 1.0, 2.0 * v / H - 1.0], dim=-1)
            s = F.grid_sample(self.maps[i][None], grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0]
            acc_all += s * wt
            w_all += wt
            wi = wt * inside.float()
            acc_in += s * wi
            w_in += wi
        mp = torch.where((w_in > 1e-6)[None], acc_in / w_in.clamp_min(1e-12),
                         torch.where((w_all > 1e-12)[None], acc_all / w_all.clamp_min(1e-12), self.binf_mean[:, None, None].to(d.device)))
        out = dict(params_D=self.params_D.clone(), params_B=self.params_B.clone(), B_inf=mp.mean((1, 2)), B_inf_map=mp)
        if self.scale is not None:
            out["beta_D_scale"] = self.scale.clone().to(d.device)
        return out

    def seen_fraction(self, c2w, focal, pp, h, w):
        """Share of the rays of a camera whose direction some map frame holds in its image."""
        d = pixel_dirs(c2w.float(), float(focal), pp, h, w)
        H, W = self.hw
        seen = torch.zeros(h, w, dtype=torch.bool, device=d.device)
        for i in range(self.maps.shape[0]):
            dc = d @ self.R[i]
            z = dc[..., 2]
            zs = torch.where(z > 1e-6, z, torch.ones_like(z))
            u = self.f[i] * dc[..., 0] / zs + self.cx[i]
            v = self.f[i] * dc[..., 1] / zs + self.cy[i]
            seen |= (z > 1e-6) & (u > 0) & (u < W) & (v > 0) & (v < H)
        return float(seen.float().mean())


def veil_residual(binf, img, known, gain, slack):
    lo = known + binf * gain
    return torch.relu(lo - img) + torch.relu(img - lo - slack)


def _samples(med, c2w, intr, idx, h, w, step, obs=None):
    """Rays (N, 3), camera centres (N, 3) and B_inf targets (N, 3) of the frames idx on a grid every step pixels."""
    D, P, T, O = [], [], [], []
    for i in idx:
        f, pp = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
        d = pixel_dirs(c2w[i], f, pp, h, w, step)
        if "B_inf_map" in med:
            mp = med["B_inf_map"][i]
            t = mp[:, step // 2::step, step // 2::step][:, :d.shape[0], :d.shape[1]].permute(1, 2, 0)
        else:
            t = med["B_inf"][i][None, None].expand(d.shape[0], d.shape[1], 3)
        if obs is not None:
            cut = lambda a_: a_[..., step // 2::step, step // 2::step][..., :d.shape[0], :d.shape[1]].permute(1, 2, 0).reshape(-1, 3)
            O.append(torch.stack([cut(obs[k][i]).to(t.dtype) for k in ("img", "known", "gain", "slack")]))
        D.append(d.reshape(-1, 3))
        P.append(c2w[i, :3, 3].expand(d.shape[0] * d.shape[1], 3))
        T.append(t.reshape(-1, 3))
    if obs is not None:
        return torch.cat(D), torch.cat(P), torch.cat(T), torch.cat(O, dim=1)
    return torch.cat(D), torch.cat(P), torch.cat(T)


def fit_medium_field(med, c2w, intr, h, w, train_idx=None, step=16, iters=1500, lr=5e-3, seed=0, obs=None, prior=0.06, **cfg):
    """Field fitted to the media med (from training/physics_splat.medium) of the frames train_idx (all by default)
    with cameras c2w (n, 4, 4) and intrinsics intr (n, 3, 3) at the image size h x w."""
    n = c2w.shape[0]
    idx = list(range(n)) if train_idx is None else [int(i) for i in train_idx]
    dev = c2w.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        field = MediumField(**cfg).to(dev)
    with torch.no_grad():
        for p in field.mlp[-1].parameters():
            p.zero_()
    c = c2w[idx, :3, 3].float()
    field.centre.copy_(c.mean(0))
    field.radius.copy_((c - c.mean(0)).norm(dim=-1).max().clamp_min(1e-6))
    field.params_D.copy_(med["params_D"][idx].float().mean(0))
    field.params_B.copy_(med["params_B"][idx].float().mean(0))
    if "beta_D_scale" in med:
        field.beta_D_scale.copy_(torch.as_tensor(med["beta_D_scale"], dtype=torch.float32).reshape(3))
        field.has_scale = True
    Ob = None
    if obs is None:
        D, P, T = _samples(med, c2w.float(), intr.float(), idx, h, w, step)
    else:
        D, P, T, Ob = _samples(med, c2w.float(), intr.float(), idx, h, w, step, obs=obs)
    field.binf_mean.copy_(T.mean(0).clamp(1e-3, 1 - 1e-3))
    opt = torch.optim.Adam(field.mlp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    perm = torch.randperm(D.shape[0], generator=g).to(dev)
    D, P, T = D[perm], P[perm], T[perm]
    if Ob is not None:
        Ob = Ob[:, perm]
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        if Ob is None:
            loss = (field(D, P) - T).abs().mean()
        else:
            pr = field(D, P)
            loss = veil_residual(pr, Ob[0], Ob[1], Ob[2], Ob[3]).square().mean() + prior * (pr - T).square().mean()
        loss.backward()
        opt.step()
        sched.step()
    field.eval()
    for p in field.parameters():
        p.requires_grad_(False)
    return field


@torch.no_grad()
def binf_error(field_or_med, med, c2w, intr, idx, h, w, step=16):
    """Mean absolute error of B_inf over the rays of the frames idx against their own maps, for a field or, with a
    callable, for the medium it returns for a frame index."""
    errs = []
    for i in idx:
        f, pp = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
        own = med["B_inf_map"][i] if "B_inf_map" in med else med["B_inf"][i][:, None, None].expand(3, h, w)
        if isinstance(field_or_med, (MediumField, ViewMedium)):
            pred = field_or_med.medium_at(c2w[i].float(), f, pp, h, w)["B_inf_map"]
        else:
            m = field_or_med(i)
            pred = m["B_inf_map"] if "B_inf_map" in m else m["B_inf"][:, None, None].expand(3, h, w)
        errs.append(float((pred.float() - own.float())[:, ::step, ::step].abs().mean()))
    return sum(errs) / max(len(errs), 1)


def held_out_check(med, c2w, intr, h, w, frame_nums, every=6, **kw):
    """Fit on all map frames but every every-th, score B_inf on the left out ones for the field and for the mean of
    the two frames nearest in frame number among the fitted ones (the medium the evaluation borrows today)."""
    n = c2w.shape[0]
    test = list(range(every // 2, n, every))
    train = [i for i in range(n) if i not in set(test)]
    field = fit_medium_field(med, c2w, intr, h, w, train_idx=train, **kw)
    nums = [int(frame_nums[i]) for i in train]

    def nearest(i):
        order = sorted(range(len(train)), key=lambda k: abs(nums[k] - int(frame_nums[i])))[:2]
        sel = [train[k] for k in order]
        out = dict(B_inf=med["B_inf"][sel].mean(0))
        if "B_inf_map" in med:
            out["B_inf_map"] = med["B_inf_map"][sel].mean(0)
        return out

    views = ViewMedium(med, c2w, intr, h, w, idx=train)
    return dict(n_left_out=len(test), field=binf_error(field, med, c2w, intr, test, h, w),
                nearest=binf_error(nearest, med, c2w, intr, test, h, w),
                views=binf_error(views, med, c2w, intr, test, h, w),
                spread_params_D=float(med["params_D"].float().std(0).mean()),
                spread_binf=float(med["B_inf"].float().std(0).mean()))
