"""Beta primitives of the feed forward map, static by default and 4D where the head says so.

Every pixel of a map frame k gives one primitive with centre mu, opacity o, Gaussian sigmas and
rotation, a Beta shape beta, and with a 4D head a dynamic share p (the gate), a temporal width w,
a temporal shape tb and a velocity v. It is split into two copies:
  static   opacity min(o (1 - p), 1), centre mu, no time. Fused in voxels with weight conf (1 - p).
  dynamic  opacity o max(p, 0) win(t - t_k), centre mu + v (t - t_k). Not fused, kept only where
           o max(p, 0) > 1/255.
win is the temporal opacity term of Universal Beta Splatting, (1 - tanh((dt / w)^2)) ^ (4 exp(tb)).
With p = 0 the static copy is the fused map of today and there is no dynamic copy.

The dictionaries keep Gaussian sigmas. K_BETA4 turns them into the support of the Beta kernel and
is applied only at the render call and in export_ubs7d, which writes the map as the 7D parameters
of ubs/scene/beta_model.py (xyz, view direction 3, time 1).
"""

import math

import numpy as np
import torch

K_BETA4 = 3.0897
EPS2D_BETA4 = 0.3 * K_BETA4 ** 2
DYN_MIN_OPACITY = 1.0 / 255.0

_BETA_REF = 4.0
_TIME_KEYS = ("t_src", "tw", "tb", "vel")


def beta_scale_factor(b=4.0):
    """(k, relative L2 residual) of the fit of (1 - d2 / k^2)^b to exp(-d2 / 2). Uses scipy.

    Least squares over the image plane, where the area element is constant in d2, as
    beta_footprint in tools/refine_gaussian_map.py.
    """
    from scipy.optimize import minimize_scalar
    d2 = np.linspace(0.0, max(80.0, 8.0 * b + 20.0), 400001)
    g = np.exp(-0.5 * d2)

    def err(k2):
        return float(((np.clip(1.0 - d2 / k2, 0.0, None) ** b - g) ** 2).sum() / (g * g).sum())

    k2 = minimize_scalar(err, bounds=(0.5, 8.0 * b + 20.0), method="bounded", options={"xatol": 1e-9}).x
    return float(math.sqrt(k2)), float(math.sqrt(err(k2)))


def frame_gaps(times):
    """times (F,) seconds -> (F,) gap of every frame.

    The gap is the larger distance to the previous and the next frame in time, the one distance for
    a frame with one neighbour, and 1.0 s when that is 0 or there is one frame. Arrays give float64.
    """
    t = times if torch.is_tensor(times) else torch.as_tensor(np.asarray(times, dtype=np.float64))
    t = t.reshape(-1)
    n = t.numel()
    if n < 2:
        return torch.ones_like(t)
    order = torch.argsort(t)
    d = t[order][1:] - t[order][:-1]
    gap = torch.maximum(torch.cat([d[:1], d]), torch.cat([d, d[-1:]]))
    gap = torch.where(gap > 0, gap, torch.ones_like(gap))
    out = torch.empty_like(t)
    out[order] = gap
    return out


def temporal_window(dt, width, tb):
    """(1 - tanh((dt / width)^2)) ^ (4 exp(tb)), broadcast, clamped as in the UBS code."""
    dt = torch.as_tensor(dt)
    d = torch.tanh((dt / width).square())
    d = d.clamp(0.0, 1.0 - torch.finfo(d.dtype).eps)
    tb = torch.as_tensor(tb, dtype=d.dtype, device=d.device)
    return (1.0 - d).pow(4.0 * torch.exp(tb))


def world_velocity(vel, fp, rot):
    """vel (N, 3) footprints per second in the source camera, fp (N,), rot (N, 3, 3) -> (N, 3) world units per second."""
    return torch.matmul(rot, (vel * fp[..., None])[..., None])[..., 0]


def split_static_dynamic(prim, fuse, voxel_px, four_d=True, dyn_max=0, extra_mean=()):
    """Static and dynamic copies of flat per pixel primitives.

    prim: (N, ...) tensors xyz, fp, conf, opacity, scales, quat, color_raw, color_clean, beta, and with
    four_d also gate, t_src, tw (seconds), tb, vel (world). fuse is voxel_fuse, voxel_px <= 0 keeps
    every row. Without four_d the gate is 0 and there is no dynamic copy. dyn_max > 0 keeps that many
    dynamic rows with the largest o p_plus.
    Returns dict(static=..., dyn=... or None, n_in, n_static, n_dyn, gate_mean).
    static: xyz, opacity, scales, quat, beta, color_raw, color_clean and the extra_mean keys.
    dyn: the same keys (opacity is o p_plus) plus t_src, tw, tb, vel.
    """
    xyz, opacity = prim["xyz"], prim["opacity"]
    n_in = xyz.shape[0]
    beta = prim.get("beta")
    if beta is None:
        beta = torch.full((n_in,), _BETA_REF, device=xyz.device, dtype=opacity.dtype)
    gate = prim["gate"] if four_d else None
    keep = 1.0 - gate if four_d else 1.0
    o_s = (opacity * keep).clamp_max(1.0)
    if voxel_px > 0:
        mean_attrs = dict(opacity=o_s, color_raw=prim["color_raw"], color_clean=prim["color_clean"])
        for k in extra_mean:
            mean_attrs[k] = prim[k]
        mean_attrs["log_beta"] = torch.log(beta / _BETA_REF)
        sx, m, p = fuse(xyz, prim["conf"] * keep, prim["fp"], mean_attrs,
                        dict(scales=prim["scales"], quat=prim["quat"]), voxel_px)
        static = dict(xyz=sx, opacity=m["opacity"], scales=p["scales"], quat=p["quat"],
                      beta=_BETA_REF * torch.exp(m["log_beta"]), color_raw=m["color_raw"], color_clean=m["color_clean"])
        for k in extra_mean:
            static[k] = m[k]
    else:
        static = dict(xyz=xyz, opacity=o_s, scales=prim["scales"], quat=prim["quat"], beta=beta,
                      color_raw=prim["color_raw"], color_clean=prim["color_clean"])
        for k in extra_mean:
            static[k] = prim[k]

    dyn = None
    if four_d:
        o_d = opacity * gate.clamp_min(0.0)
        score = o_d.detach()
        idx = torch.nonzero(score > DYN_MIN_OPACITY, as_tuple=False)[:, 0]
        if dyn_max > 0 and idx.numel() > dyn_max:
            top = torch.topk(score[idx], int(dyn_max), sorted=False).indices
            idx = idx[torch.sort(top).values]
        if idx.numel() > 0:
            dyn = dict(xyz=xyz[idx], opacity=o_d[idx], scales=prim["scales"][idx], quat=prim["quat"][idx],
                       beta=beta[idx], color_raw=prim["color_raw"][idx], color_clean=prim["color_clean"][idx])
            for k in tuple(extra_mean) + _TIME_KEYS:
                dyn[k] = prim[k][idx]
    return dict(static=static, dyn=dyn, n_in=n_in, n_static=int(static["xyz"].shape[0]),
                n_dyn=0 if dyn is None else int(dyn["xyz"].shape[0]),
                gate_mean=float(gate.detach().float().mean()) if four_d and n_in > 0 else 0.0)


def primitives_at_time(g, t=None, min_opacity=0.0):
    """Flat dict xyz, opacity, scales, quat, beta, color_raw, color_clean (and extras) for one query time.

    Static rows first, then the dynamic rows moved and faded to t. With t None or no dynamic rows it
    returns g['static'] itself. min_opacity > 0 leaves out dynamic rows whose opacity at t is not above it.
    """
    st, dyn = g["static"], g.get("dyn")
    if t is None or dyn is None or dyn["xyz"].shape[0] == 0:
        return st
    t_src = dyn["t_src"]
    dt = torch.as_tensor(t, dtype=t_src.dtype, device=t_src.device) - t_src
    o_t = dyn["opacity"] * temporal_window(dt, dyn["tw"], dyn["tb"])
    moved = dict(dyn, opacity=o_t, xyz=dyn["xyz"] + dyn["vel"] * dt[:, None])
    if min_opacity > 0:
        idx = torch.nonzero(o_t.detach() > min_opacity, as_tuple=False)[:, 0]
        moved = {k: moved[k][idx] for k in st}
    return {k: torch.cat([st[k], moved[k]]) for k in st}


def rescale(g, unit):
    """Copy of g with xyz, scales and vel multiplied by unit. g is a split or a flat dict."""
    def one(d):
        if d is None:
            return None
        out = dict(d)
        for k in ("xyz", "scales", "vel"):
            if k in out:
                out[k] = out[k] * unit
        return out

    if "static" in g:
        return dict(g, static=one(g["static"]), dyn=one(g.get("dyn")))
    return one(g)


def _quat_to_rotmat(q):
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


_IU = ((0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2))
_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
_SIGNS = tuple((x, y, z) for x in (1.0, -1.0) for y in (1.0, -1.0) for z in (1.0, -1.0))
_W6 = (1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0))


def _skew_gen(dtype, device):
    """(3, 3, 3) generators of A, A[0,1] = a0, A[0,2] = a1, A[1,2] = a2, A skew (UWBS l_triangle[:, :3])."""
    e = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for m, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        e[m, i, j], e[m, j, i] = 1.0, -1.0
    return e


def _ubs_factor(a, gen):
    return torch.eye(3, dtype=a.dtype, device=a.device) + torch.einsum("nm,mij->nij", a, gen)


def _lm(p, cov, nrm, gen, iters, tol):
    """Levenberg-Marquardt on p = (a, log s) for (I + A) diag(s)^2 (I + A)^T = cov. Returns p and the relative error."""
    iu0, iu1 = _IU
    w6 = cov.new_tensor(_W6)
    tgt = cov[:, iu0, iu1]

    def resid(p):
        b = _ubs_factor(p[:, :3], gen)
        e = torch.exp(2.0 * p[:, 3:])
        m = b * e[:, None, :]
        f = m @ b.transpose(1, 2)
        return (f[:, iu0, iu1] - tgt) * w6 / nrm[:, None], b, m, e

    r, b, m, e = resid(p)
    lam = torch.full((p.shape[0], 1, 1), 1e-3, dtype=p.dtype, device=p.device)
    for it in range(iters):
        x = torch.einsum("mij,nkj->nmik", gen, m)
        d_a = x + x.transpose(-1, -2)
        bt = b.transpose(1, 2)
        d_l = 2.0 * e[:, :, None, None] * bt[:, :, :, None] * bt[:, :, None, :]
        jac = (torch.cat([d_a, d_l], dim=1)[:, :, iu0, iu1] * w6 / nrm[:, None, None]).transpose(1, 2)
        h = jac.transpose(1, 2) @ jac
        grad = jac.transpose(1, 2) @ r[..., None]
        damp = lam * torch.diag_embed(torch.diagonal(h, dim1=1, dim2=2) + 1e-12)
        step = torch.linalg.solve(h + damp, grad)[..., 0]
        pn = p - step
        rn, bn, mn, en = resid(pn)
        ok = torch.isfinite(rn).all(-1) & (rn.norm(dim=-1) < r.norm(dim=-1))
        p = torch.where(ok[:, None], pn, p)
        r = torch.where(ok[:, None], rn, r)
        b = torch.where(ok[:, None, None], bn, b)
        m = torch.where(ok[:, None, None], mn, m)
        e = torch.where(ok[:, None], en, e)
        lam = torch.where(ok[:, None, None], lam * 0.3, lam * 10.0).clamp(1e-12, 1e8)
        if it % 5 == 4 and bool((r.norm(dim=-1) < tol).all()):
            break
    return p, r.norm(dim=-1)


def _eigen_starts(cov, gen):
    """Starts (n, 48, 6) from the eigenvector frames of cov in every axis order and sign, and the cos of their
    rotation angle (n, 48), -inf for the 24 frames with determinant -1."""
    lam, u = torch.linalg.eigh(cov)
    starts, cosang = [], []
    for perm in _PERMS:
        for sg in _SIGNS:
            up = u[:, :, list(perm)] * cov.new_tensor(sg)
            proper = torch.linalg.det(up) > 0
            cos = ((torch.diagonal(up, dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            a = torch.stack([up[:, 0, 1] - up[:, 1, 0], up[:, 0, 2] - up[:, 2, 0], up[:, 1, 2] - up[:, 2, 1]], -1)
            a = a / (2.0 * cos.abs().clamp_min(0.2))[:, None]
            nb = _ubs_factor(a, gen).square().sum(1)
            ls = 0.5 * torch.log(lam[:, list(perm)].clamp_min(1e-300) / nb)
            starts.append(torch.cat([a, ls], -1))
            cosang.append(torch.where(proper, cos, torch.full_like(cos, -math.inf)))
    return torch.stack(starts, 1), torch.stack(cosang, 1)


def covariance_to_ubs(cov, iters=40, chunk=262144, tol=1.0e-10):
    """cov (N, 3, 3) -> scales (N, 3), a (N, 3), rel_err (N,) with (I + A) diag(s)^2 (I + A)^T ~ cov.

    rel_err is the relative Frobenius error. Levenberg-Marquardt on (a, log s) in float64, first from
    a = 0, then for the rows not met to tol from the 24 proper eigenvector frames of cov, the frame
    closest to the identity first. The best result of every row is kept. Results in the dtype of cov.
    """
    dtype = cov.dtype
    c64 = cov.to(torch.float64)
    n = c64.shape[0]
    gen = _skew_gen(torch.float64, c64.device)
    s_out = torch.full((n, 3), math.nan, dtype=torch.float64, device=c64.device)
    a_out = torch.zeros(n, 3, dtype=torch.float64, device=c64.device)
    e_out = torch.full((n,), math.inf, dtype=torch.float64, device=c64.device)
    finite = torch.isfinite(c64).flatten(1).all(1)
    rows = torch.nonzero(finite, as_tuple=False)[:, 0]
    for c0 in range(0, rows.numel(), chunk):
        ri = rows[c0:c0 + chunk]
        c = 0.5 * (c64[ri] + c64[ri].transpose(1, 2))
        nrm = c.flatten(1).norm(dim=1).clamp_min(1e-300)
        p0 = torch.cat([torch.zeros(ri.numel(), 3, dtype=torch.float64, device=c.device),
                        0.5 * torch.log(torch.diagonal(c, dim1=1, dim2=2).clamp_min(1e-300))], -1)
        best_p, best_e = _lm(p0, c, nrm, gen, iters, tol)
        todo = best_e > tol
        if bool(todo.any()):
            starts, cosang = _eigen_starts(c, gen)
            order = torch.argsort(cosang, dim=1, descending=True)
            for k in range(24):
                sel = torch.nonzero(todo, as_tuple=False)[:, 0]
                if sel.numel() == 0:
                    break
                ok = torch.isfinite(cosang[sel, order[sel, k]])
                sel = sel[ok]
                if sel.numel() == 0:
                    continue
                p, e = _lm(starts[sel, order[sel, k]], c[sel], nrm[sel], gen, iters, tol)
                better = e < best_e[sel]
                best_p[sel] = torch.where(better[:, None], p, best_p[sel])
                best_e[sel] = torch.where(better, e, best_e[sel])
                todo = best_e > tol
        s_out[ri] = torch.exp(best_p[:, 3:])
        a_out[ri] = best_p[:, :3]
        e_out[ri] = best_e
    return s_out.to(dtype), a_out.to(dtype), e_out.to(dtype)


def _inverse_softplus(y):
    return y + torch.log(-torch.expm1(-y))


def export_ubs7d(g, t_range, color="color_clean", view_scale=1.0e3, static_width=1.0e3, k_beta=K_BETA4):
    """g of split_static_dynamic in SfM units, t_range (t0, t1) seconds of the map frames (a range of
    length 0 counts as 1 s).

    Returns a dict of float32 tensors with N = n_static + n_dyn rows, static rows first:
      xyz (N, 3), mean (N, 4), scale (N, 7), l_triangle (N, 21), rgb (N, 3), opacity (N, 1), beta (N, 5),
    all as the raw parameters of ubs/scene/beta_model.py with input_dim 7, plus 'fit_err' (N,), the
    relative error of the spatial factor, and 'clamped' (N,) bool, dynamic rows whose velocity was
    scaled down because the 7D factor could not hold it. Colours are clamped at 0 as in render_view_beta.
    """
    f64 = torch.float64
    st, dyn = g["static"], g.get("dyn")
    parts = [st] if dyn is None or dyn["xyz"].shape[0] == 0 else [st, dyn]
    n_s = int(st["xyz"].shape[0])

    def cat(k):
        return torch.cat([q[k].reshape(q[k].shape[0], -1) for q in parts]).to(f64)

    xyz, sig, quat = cat("xyz"), cat("scales"), cat("quat")
    o, rgb = cat("opacity")[:, 0], cat(color)
    beta = cat("beta")[:, 0] if "beta" in st else torch.full_like(o, _BETA_REF)
    n = xyz.shape[0]
    dev = xyz.device
    rot = _quat_to_rotmat(quat)
    cov_b = (rot * (k_beta * sig)[:, None, :].square()) @ rot.transpose(1, 2)
    t0, t1 = float(t_range[0]), float(t_range[1])
    span = t1 - t0 if t1 > t0 else 1.0

    mean = torch.zeros(n, 4, dtype=f64, device=dev)
    mean[:, 3] = 0.5
    t_scale = torch.full((n,), float(static_width), dtype=f64, device=dev)
    t_beta = torch.zeros(n, dtype=f64, device=dev)
    ell = torch.zeros(n, 3, dtype=f64, device=dev)
    clamped = torch.zeros(n, dtype=torch.bool, device=dev)
    v11 = cov_b.clone()
    if len(parts) > 1:
        w = dyn["tw"].to(f64) / span
        tb = dyn["tb"].to(f64)
        badj = torch.exp(tb).clamp_max(1.0)
        c = dyn["vel"].to(f64) * span * (w.square() / badj)[:, None]
        sd = cov_b[n_s:]
        # with an exact factor |ell|^2 = q / (1 + badj q / W^2), q = c^T Sigma_b^-1 c, so W^2 > |ell|^2 iff q (1 - badj) < W^2
        q = (c * torch.linalg.solve(sd, c[..., None])[..., 0]).sum(-1)
        bad = q * (1.0 - badj) >= w.square()
        lam = torch.sqrt(0.99 * w.square() / ((1.0 - 0.99 * badj) * q).clamp_min(1e-300))
        c = torch.where(bad[:, None], c * lam[:, None], c)
        v11[n_s:] = sd + (badj / w.square())[:, None, None] * c[:, :, None] * c[:, None, :]
        mean[n_s:, 3] = (dyn["t_src"].to(f64) - t0) / span
        t_beta[n_s:] = tb
        clamped[n_s:] = bad
    s_xyz, a, fit_err = covariance_to_ubs(v11)
    if len(parts) > 1:
        lx = _ubs_factor(a[n_s:], _skew_gen(f64, dev)) * s_xyz[n_s:, None, :]
        el = torch.linalg.solve(lx, c[..., None])[..., 0]
        el2 = el.square().sum(-1)
        over = ~(el2 < w.square() * (1.0 - 1e-6))
        el = torch.where(over[:, None], el * torch.sqrt(0.99 * w.square() / el2.clamp_min(1e-300))[:, None], el)
        ell[n_s:] = el
        t_scale[n_s:] = torch.sqrt((w.square() - el.square().sum(-1)).clamp_min(0.0))
        clamped[n_s:] |= over

    scale = torch.cat([s_xyz, torch.full((n, 3), float(view_scale), dtype=f64, device=dev), t_scale[:, None]], -1)
    l_tri = torch.zeros(n, 21, dtype=f64, device=dev)
    l_tri[:, 0:3] = a
    l_tri[:, 15:18] = ell
    oc = o.clamp(1e-4, 1.0 - 1e-4)
    betas = torch.cat([torch.log(beta / _BETA_REF)[:, None], torch.full((n, 3), -3.0, dtype=f64, device=dev),
                       t_beta[:, None]], -1)
    out = dict(xyz=xyz, mean=mean, scale=_inverse_softplus(scale), l_triangle=l_tri, rgb=rgb.clamp_min(0.0),
               opacity=torch.log(oc / (1.0 - oc))[:, None], beta=betas, fit_err=fit_err)
    out = {k: v.float() for k, v in out.items()}
    out["clamped"] = clamped
    fe = fit_err.float().cpu().numpy()
    fe = fe[np.isfinite(fe)]
    if fe.size:
        print(f"[ubs7d] {n_s} static and {n - n_s} dynamic rows, fit_err median {np.median(fe):.2e} "
              f"p99 {np.percentile(fe, 99):.2e} max {fe.max():.2e}, {int(clamped.sum())} velocities clamped", flush=True)
    return out


def _ubs_ply_columns(n_beta=5, n_mean=4, n_scale=7, n_tri=21):
    """Column names of BetaModel.save_ply, in order."""
    return (["x", "y", "z", "red", "green", "blue", "opacity"] + [f"beta_{i}" for i in range(n_beta)]
            + [f"mean_{i}" for i in range(n_mean)] + [f"scale_{i}" for i in range(n_scale)]
            + [f"l_triangle_{i}" for i in range(n_tri)])


def write_ubs_ply(path, params):
    """PLY with the columns of BetaModel.save_ply: x y z red green blue opacity beta_0..4 mean_0..3 scale_0..6 l_triangle_0..20."""
    blocks = [params[k].detach().reshape(params[k].shape[0], -1).float().cpu().numpy()
              for k in ("xyz", "rgb", "opacity", "beta", "mean", "scale", "l_triangle")]
    names = _ubs_ply_columns(blocks[3].shape[1], blocks[4].shape[1], blocks[5].shape[1], blocks[6].shape[1])
    data = np.concatenate(blocks, axis=1).astype("<f4")
    assert data.shape[1] == len(names), (data.shape, len(names))
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {data.shape[0]}"]
    head += [f"property float {k}" for k in names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
