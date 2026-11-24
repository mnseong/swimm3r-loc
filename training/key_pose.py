"""Key pose of a frame against the anchor's own Gaussian centres, and the homing hinge.

The key pose is the pose at which the map of the Gaussian branch is registered in a frame.
Reciprocal descriptor matches between the frame and the anchor of the same pair pass tie pixels of
the frame to the centres of the anchor's Gaussians, and a robust Gauss-Newton PnP moves the
Procrustes pose onto them. The pose is a constant for the head loss (docs/beta4d_keyhome_spec.md,
section 6). The homing hinge asks the photometric registration loss of the map to rise away from
the key pose at least as much as the target image itself changes.

Poses are camera to world (4, 4), the camera looks along +z, the centre of pixel j is at j + 0.5
and pp is the principal point (x, y) in the same continuous pixel coordinates.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BEHIND_PX = 1.0e4  # reprojection error given to a point behind the camera
TRIM_ROUNDS = 3


@dataclass
class KeyPoseConfig:
    stride: int = 8
    max_iter: int = 10
    max_matches: int = 4096
    iters: int = 10
    huber_px: float = 2.0
    inlier_px: float = 2.0
    min_matches: int = 200
    min_inlier: float = 0.3
    max_move_px: float = 40.0
    trim_px: float = 0.0  # above 0: after the Huber fit, TRIM_ROUNDS refits on the matches within trim_px
    gate_px: float = 0.0  # above 0: only matches within gate_px of their projection at the start pose (motion prior)
    min_score: float = -1.0  # matches with a descriptor dot product under min_score are dropped


def _skew(v):
    """(..., 3) -> (..., 3, 3) with skew(v) x = v cross x."""
    o = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(-1)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(*v.shape[:-1], 3, 3)


def _rot_log(r):
    """Rotation vector (3,) of a rotation matrix (3, 3), through the quaternion, stable near 0 and pi."""
    m00, m11, m22 = r[0, 0], r[1, 1], r[2, 2]
    k = int(torch.stack([m00 + m11 + m22, m00, m11, m22]).argmax())
    if k == 0:
        w = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(0.0))
        x, y, z = (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)
    elif k == 1:
        x = 0.5 * torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0))
        w, y, z = (r[2, 1] - r[1, 2]) / (4 * x), (r[0, 1] + r[1, 0]) / (4 * x), (r[0, 2] + r[2, 0]) / (4 * x)
    elif k == 2:
        y = 0.5 * torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(0.0))
        w, x, z = (r[0, 2] - r[2, 0]) / (4 * y), (r[0, 1] + r[1, 0]) / (4 * y), (r[1, 2] + r[2, 1]) / (4 * y)
    else:
        z = 0.5 * torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(0.0))
        w, x, y = (r[1, 0] - r[0, 1]) / (4 * z), (r[0, 2] + r[2, 0]) / (4 * z), (r[1, 2] + r[2, 1]) / (4 * z)
    v = torch.stack([x, y, z])
    if float(w) < 0:
        w, v = -w, -v
    s = v.norm()
    if float(s) < 1e-12:
        return v * (2.0 / w)
    return v * (2.0 * torch.atan2(s, w) / s)


def _rigid_inv(c2w):
    rt = c2w[:3, :3].transpose(0, 1)
    top = torch.cat([rt, -(rt @ c2w[:3, 3])[:, None]], dim=1)
    return torch.cat([top, c2w[3:]], dim=0)


def _as_pp(pp, like):
    return torch.as_tensor(pp, dtype=like.dtype, device=like.device).reshape(2)


def delta_pose(xi):
    """xi (6,) rotation vector then translation -> (4, 4)."""
    top = torch.cat([torch.linalg.matrix_exp(_skew(xi[:3])), xi[3:, None]], dim=1)
    return torch.cat([top, xi.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)


def delta_from_poses(c2w_a, c2w_b):
    """xi with c2w_b = c2w_a @ delta_pose(xi), computed in float64, returned in the dtype of c2w_a."""
    d = _rigid_inv(c2w_a.double()) @ c2w_b.double()
    return torch.cat([_rot_log(d[:3, :3]), d[:3, 3]]).to(c2w_a.dtype)


def project(xyz, c2w, focal, pp):
    """World points (M, 3) -> pixels (M, 2) and depth (M,) in the camera c2w (4, 4)."""
    c2w = c2w.to(xyz)
    xc = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    z = xc[:, 2]
    return focal * xc[:, :2] / z[:, None] + _as_pp(pp, xyz), z


@torch.no_grad()
def _nn_dot(q, db, chunk):
    """Index and value of the largest dot product with the rows of db (N, D), for every row of q (M, D)."""
    m = q.shape[0]
    best = torch.full((m,), -math.inf, device=q.device, dtype=q.dtype)
    idx = torch.zeros(m, dtype=torch.long, device=q.device)
    block = 16 * chunk
    for s in range(0, m, chunk):
        qs, bv, bi = q[s:s + chunk], best[s:s + chunk], idx[s:s + chunk]
        for b in range(0, db.shape[0], block):
            v, j = (qs @ db[b:b + block].transpose(0, 1)).max(dim=1)
            up = v > bv
            bv = torch.where(up, v, bv)
            bi = torch.where(up, j + b, bi)
        best[s:s + chunk], idx[s:s + chunk] = bv, bi
    return idx, best


@torch.no_grad()
def reciprocal_matches(desc_a, desc_b, stride=8, max_iter=10, max_matches=4096, chunk=2048):
    """Mutual nearest neighbours by dot product, the scheme of mast3r/fast_nn.py:fast_reciprocal_NNs.

    desc_a, desc_b (H, W, D). The start points are the pixels of a on a grid of stride (offset
    stride // 2). Nearest neighbour in b, then back in a, repeated until the pair is a fixed point
    or max_iter is reached. Unconverged starts are dropped and duplicates removed. The matches are
    sorted by score (the dot product), at most max_matches are kept (0 keeps all).
    Returns xy_a (M, 2), xy_b (M, 2) pixel centres (u + 0.5, v + 0.5), score (M,).
    """
    ha, wa, d = desc_a.shape
    hb, wb, _ = desc_b.shape
    fa = desc_a.reshape(-1, d)
    fb = desc_b.reshape(-1, d).to(fa)
    dev = fa.device
    ys = torch.arange(stride // 2, ha, stride, device=dev)
    xs = torch.arange(stride // 2, wa, stride, device=dev)
    ia = (ys[:, None] * wa + xs[None, :]).reshape(-1)
    ib = torch.full_like(ia, -1)
    old_a, old_b = ia.clone(), ib.clone()
    todo = torch.ones_like(ia, dtype=torch.bool)
    for _ in range(max_iter):
        sel = todo.nonzero().squeeze(1)
        ib[sel] = _nn_dot(fa[ia[sel]], fb, chunk)[0]
        todo &= ib != old_b
        sel = todo.nonzero().squeeze(1)
        ia[sel] = _nn_dot(fb[ib[sel]], fa, chunk)[0]
        todo &= ia != old_a
        if not bool(todo.any()):
            break
        old_a.copy_(ia)
        old_b.copy_(ib)
    nb = hb * wb
    key = torch.unique(ia[~todo] * nb + ib[~todo])
    ia, ib = key // nb, key % nb
    score = (fa[ia] * fb[ib]).sum(-1)
    order = torch.sort(score, descending=True, stable=True).indices
    if max_matches > 0:
        order = order[:max_matches]
    ia, ib, score = ia[order], ib[order], score[order]
    xy_a = torch.stack([ia % wa, ia // wa], dim=-1).float() + 0.5
    xy_b = torch.stack([ib % wb, ib // wb], dim=-1).float() + 0.5
    return xy_a, xy_b, score


def _huber(e, k):
    return torch.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k))


def _median(e, ok):
    return float(e[ok].median()) if bool(ok.any()) else math.nan


@torch.no_grad()
def pnp_gauss_newton(xyz, uv, c2w0, focal, pp, weights=None, iters=10, huber_px=2.0, damping=1.0e-6):
    """Robust PnP from c2w0: Gauss-Newton with Huber weights (iteratively reweighted) in float64.

    Minimises the weighted mean Huber cost of |project(xyz, c2w) - uv| over c2w <- c2w @ delta_pose(xi).
    The damping is relative to the diagonal of the normal matrix, a step that raises the cost is
    halved up to 5 times, and the iteration stops when no step lowers it. A point behind the camera
    counts as an error of BEHIND_PX. Returns c2w (4, 4) in the dtype of c2w0 and
    dict(cost0, cost, resid0_px, resid_px, steps), resid the median error of the points in front.
    """
    x, p = xyz.detach().double(), uv.detach().double()
    w0 = torch.ones_like(x[:, 0]) if weights is None else weights.detach().double().clamp_min(0.0)
    keep = torch.isfinite(x).all(-1) & torch.isfinite(p).all(-1) & torch.isfinite(w0)
    x, p, w0 = x[keep], p[keep], w0[keep]
    wsum = w0.sum().clamp_min(1e-12)
    ppt = _as_pp(pp, x)
    f = float(focal)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)

    def evaluate(t):
        xc = (x - t[:3, 3]) @ t[:3, :3]
        ok = xc[:, 2] > 1e-6
        z = torch.where(ok, xc[:, 2], torch.ones_like(xc[:, 2]))
        r = f * xc[:, :2] / z[:, None] + ppt - p
        e = torch.where(ok, r.norm(dim=-1), torch.full_like(z, BEHIND_PX))
        return float((w0 * _huber(e, huber_px)).sum() / wsum), xc, z, r, e, ok

    t = c2w0.detach().double()
    cur = evaluate(t)
    cost0, resid0 = cur[0], _median(cur[4], cur[5])
    steps = 0
    for _ in range(iters):
        cost, xc, z, r, e, ok = cur
        if not math.isfinite(cost) or x.shape[0] == 0:
            break
        wr = w0 * torch.where(e <= huber_px, torch.ones_like(e), huber_px / e.clamp_min(1e-12)) * ok
        zi = 1.0 / z
        zero = torch.zeros_like(zi)
        jp = f * torch.stack([zi, zero, -xc[:, 0] * zi * zi, zero, zi, -xc[:, 1] * zi * zi], dim=-1).reshape(-1, 2, 3)
        jac = jp @ torch.cat([_skew(xc), -eye.expand(xc.shape[0], 3, 3)], dim=-1)
        jw = jac * wr[:, None, None]
        h = torch.einsum("mki,mkj->ij", jw, jac)
        g = torch.einsum("mki,mk->i", jw, r)
        h = h + damping * torch.diag(h.diagonal().clamp_min(1e-12))
        delta = -torch.linalg.solve(h, g)
        if not bool(torch.isfinite(delta).all()) or float(delta.abs().max()) == 0.0:
            break
        step, nxt = 1.0, None
        for _ in range(6):
            cand = t @ delta_pose(step * delta)
            out = evaluate(cand)
            if out[0] < cost:
                nxt = (cand, out)
                break
            step *= 0.5
        if nxt is None:
            break
        t, cur = nxt
        steps += 1
    return t.to(c2w0.dtype), dict(cost0=cost0, cost=cur[0], resid0_px=resid0, resid_px=_median(cur[4], cur[5]),
                                  steps=steps)


@torch.no_grad()
def key_pose(desc_t, desc_a, xyz_a, valid_a, c2w0, focal, pp, cfg=KeyPoseConfig(), conf_a=None):
    """Key pose of a frame t against the anchor of its pair pass.

    desc_t, desc_a (H, W, D) descriptors of the frame and of the anchor from the same pass, xyz_a
    (H, W, 3) the centres of the anchor's Gaussians in the world, valid_a (H, W), c2w0 the
    Procrustes pose. conf_a (H, W), optional, weights every match by the confidence of its anchor pixel
    in the PnP (the weights of the track loss of training/track_losses.py), none weighs all matches
    alike. Returns c2w (4, 4), accepted (bool) and stats with n_match, inlier0, inlier,
    resid0_px, resid_px, move_px, rot_deg, trans, steps, n_trim (matches of the last fit), reason.
    When not accepted c2w is c2w0. The statistics are over all matches.
    """
    h, w = valid_a.shape
    xy_t, xy_a, score = reciprocal_matches(desc_t, desc_a, cfg.stride, cfg.max_iter, max_matches=0)
    ia = xy_a[:, 1].long() * w + xy_a[:, 0].long()
    xyz = xyz_a.reshape(-1, 3)[ia]
    wt = None if conf_a is None else conf_a.reshape(-1)[ia].to(xyz)
    keep = valid_a.reshape(-1)[ia].bool() & torch.isfinite(xyz).all(-1) & (score >= cfg.min_score)
    if wt is not None:
        keep &= torch.isfinite(wt) & (wt > 0)
    xy_t, xyz = xy_t[keep], xyz[keep]
    wt = None if wt is None else wt[keep]
    if cfg.gate_px > 0 and xy_t.shape[0] > 0:
        uv_g, z_g = project(xyz.double(), c2w0.double(), focal, pp)
        near = (z_g > 1e-6) & ((uv_g - xy_t.double()).norm(dim=-1) < cfg.gate_px)
        if int(near.sum()) >= 6:
            xy_t, xyz = xy_t[near], xyz[near]
            wt = None if wt is None else wt[near]
    if cfg.max_matches > 0:
        xy_t, xyz = xy_t[:cfg.max_matches], xyz[:cfg.max_matches]
        wt = None if wt is None else wt[:cfg.max_matches]
    n = int(xy_t.shape[0])
    stats = dict(n_match=n, inlier0=0.0, inlier=0.0, resid0_px=math.nan, resid_px=math.nan, move_px=0.0,
                 rot_deg=0.0, trans=0.0, steps=0, n_trim=0, reason="few_matches")
    if n < 6:
        return c2w0, False, stats

    x64, p64 = xyz.double(), xy_t.double()
    w64 = None if wt is None else wt.double()
    c2w, st = pnp_gauss_newton(x64, p64, c2w0.double(), focal, pp, weights=w64, iters=cfg.iters, huber_px=cfg.huber_px)
    steps, n_trim = st["steps"], n
    for _ in range(TRIM_ROUNDS if cfg.trim_px > 0 else 0):
        uv, z = project(x64, c2w, focal, pp)
        sel = (z > 1e-6) & ((uv - p64).norm(dim=-1) < cfg.trim_px)
        if int(sel.sum()) < 6:
            break
        c2w, st = pnp_gauss_newton(x64[sel], p64[sel], c2w, focal, pp, weights=None if w64 is None else w64[sel],
                                   iters=cfg.iters, huber_px=cfg.huber_px)
        steps, n_trim = steps + st["steps"], int(sel.sum())
    c2w = c2w.to(c2w0.dtype)
    uv0, z0 = project(x64, c2w0.double(), focal, pp)
    uv1, z1 = project(x64, c2w.double(), focal, pp)
    e0 = torch.where(z0 > 1e-6, (uv0 - p64).norm(dim=-1), torch.full_like(z0, math.inf))
    e1 = torch.where(z1 > 1e-6, (uv1 - p64).norm(dim=-1), torch.full_like(z1, math.inf))
    front = (z0 > 1e-6) & (z1 > 1e-6)
    d = _rigid_inv(c2w0.double()) @ c2w.double()
    stats.update(
        inlier0=float((e0 < cfg.inlier_px).double().mean()), inlier=float((e1 < cfg.inlier_px).double().mean()),
        resid0_px=_median(e0, torch.isfinite(e0)), resid_px=_median(e1, torch.isfinite(e1)),
        move_px=float((uv1 - uv0).norm(dim=-1)[front].mean()) if bool(front.any()) else math.inf,
        rot_deg=math.degrees(float(_rot_log(d[:3, :3]).norm())), trans=float(d[:3, 3].norm()), steps=steps,
        n_trim=n_trim)

    finite = bool(torch.isfinite(c2w).all()) and all(
        math.isfinite(stats[k]) for k in ("inlier0", "inlier", "resid_px", "move_px", "rot_deg", "trans"))
    if not finite:
        reason = "nonfinite"
    elif n < cfg.min_matches:
        reason = "few_matches"
    elif stats["inlier"] < cfg.min_inlier:
        reason = "low_inlier"
    elif stats["inlier"] < stats["inlier0"]:
        reason = "inlier_drop"
    elif stats["move_px"] > cfg.max_move_px:
        reason = "move"
    else:
        reason = "ok"
    stats["reason"] = reason
    accepted = reason == "ok"
    return (c2w if accepted else c2w0), accepted, stats


def interp_pose(c2w, times, t):
    """c2w (K, 4, 4), times (K,) -> pose at t, slerp and linear, clamped to the range."""
    ts = torch.as_tensor(times, dtype=torch.float64).reshape(-1).cpu()
    order = torch.sort(ts, stable=True).indices
    ts = ts[order]
    t = float(t)
    if ts.numel() == 1 or t <= float(ts[0]):
        return c2w[int(order[0])].clone()
    if t >= float(ts[-1]):
        return c2w[int(order[-1])].clone()
    i = int(torch.searchsorted(ts, torch.tensor([t], dtype=torch.float64), right=True)[0]) - 1
    i = min(max(i, 0), ts.numel() - 2)
    a = (t - float(ts[i])) / (float(ts[i + 1]) - float(ts[i]))
    p0, p1 = c2w[int(order[i])].double(), c2w[int(order[i + 1])].double()
    rot = p0[:3, :3] @ torch.linalg.matrix_exp(_skew(a * _rot_log(p0[:3, :3].transpose(0, 1) @ p1[:3, :3])))
    top = torch.cat([rot, ((1.0 - a) * p0[:3, 3] + a * p1[:3, 3])[:, None]], dim=1)
    return torch.cat([top, p0[3:]], dim=0).to(c2w.dtype)


def flow_of_delta(xi, depth_med, focal, h, w):
    """Mean image flow in px of delta_pose(xi) at one depth.

    The points on the rays of a grid of pixels (every 8th, principal point at the image centre) at
    depth depth_med are seen from the camera moved by delta_pose(xi). Points that end up behind it
    are left out.
    """
    xi = xi.detach().double()
    dev = xi.device
    v, u = torch.meshgrid(torch.arange(4, h, 8, device=dev, dtype=torch.float64) + 0.5,
                          torch.arange(4, w, 8, device=dev, dtype=torch.float64) + 0.5, indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)
    x = torch.stack([(u - 0.5 * w) / focal, (v - 0.5 * h) / focal, torch.ones_like(u)], dim=-1) * float(depth_med)
    uv, z = project(x, delta_pose(xi), focal, (0.5 * w, 0.5 * h))
    ok = z > 1e-6
    if not bool(ok.any()):
        return math.inf
    return float((uv - torch.stack([u, v], dim=-1)).norm(dim=-1)[ok].mean())


def _scale_to_flow(xi, target, flow, iters=20):
    for _ in range(iters):
        f = flow(xi)
        if not (math.isfinite(f) and f > 0) or abs(f - target) <= 1e-6 * target:
            break
        xi = xi * (target / f)
    return xi


def homing_perturbations(c2w_key, c2w_int, depth_med, focal, h, w, px_max, gen):
    """List of four xi: xi_0 = delta_from_poses(c2w_key, c2w_int), xi_0 / 2 and two random ones.

    xi_0 is scaled down to a mean flow of px_max when it has more. The random directions mix rotation
    and translation at equal flow per unit and are scaled to a mean flow uniform in [0.5, px_max].
    gen is a torch.Generator on the CPU. The xi are in the dtype and device of c2w_key.
    """
    def flow(x):
        return flow_of_delta(x, depth_med, focal, h, w)

    xi0 = delta_from_poses(c2w_key.double(), c2w_int.double())
    if flow(xi0) > px_max:
        xi0 = _scale_to_flow(xi0, px_max, flow)
    out = [xi0, 0.5 * xi0]
    lo = min(0.5, px_max)
    for _ in range(2):
        target = lo + (px_max - lo) * float(torch.rand((), generator=gen, dtype=torch.float64))
        d = torch.randn(6, generator=gen, dtype=torch.float64)
        d = torch.cat([d[:3] / focal, d[3:] * float(depth_med) / focal]).to(xi0.device)
        out.append(_scale_to_flow(d, target, flow))
    return [x.to(dtype=c2w_key.dtype, device=c2w_key.device) for x in out]


@torch.no_grad()
def warp_by_pose(img, depth, focal, pp, xi):
    """img (3, H, W), depth (H, W) of the same camera -> the image seen after delta_pose(xi), and a valid mask.

    Backward warp in float64: a pixel of the moved camera takes the depth of the same pixel of the
    source, its point is projected into the source camera and the image is sampled there
    (bilinear). Exact for a rotation, first order in the depth change for a translation. Pixels
    whose sample falls outside the image, or without a positive depth, are 0 and not valid.
    """
    _, h, w = img.shape
    dev = img.device
    d = delta_pose(xi.detach().to(device=dev, dtype=torch.float64))
    z = depth.detach().double()
    cx, cy = (float(c) for c in pp)
    v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float64) + 0.5,
                          torch.arange(w, device=dev, dtype=torch.float64) + 0.5, indexing="ij")
    xn = torch.stack([(u - cx) / focal, (v - cy) / focal, torch.ones_like(u)], dim=-1) * z[..., None]
    xs = xn @ d[:3, :3].transpose(0, 1) + d[:3, 3]
    ok = torch.isfinite(z) & (z > 1e-4) & torch.isfinite(xs).all(-1) & (xs[..., 2] > 1e-4)
    zs = torch.where(ok, xs[..., 2], torch.ones_like(z))
    us, vs = focal * xs[..., 0] / zs + cx, focal * xs[..., 1] / zs + cy
    ok &= (us >= 0.5) & (us <= w - 0.5) & (vs >= 0.5) & (vs <= h - 0.5)
    grid = torch.stack([2.0 * us / w - 1.0, 2.0 * vs / h - 1.0], dim=-1)
    grid = torch.where(ok[..., None], grid, torch.zeros_like(grid))
    out = F.grid_sample(img.detach().double()[None], grid[None], mode="bilinear", padding_mode="border",
                        align_corners=False)[0]
    return (out * ok).to(img.dtype), ok


def homing_loss(render_fn, c2w_key, xis, target, depth, focal, pp, kappa=0.7, eps=1.0e-3):
    """render_fn(c2w) -> (3, H, W). Returns the loss and the share of violated perturbations.

    l(T) = mean |render_fn(T) - target|, a_s = mean |warp_by_pose(target, xi_s) - target| over the
    valid pixels, loss = mean_s relu(kappa a_s - (l(T_key D(xi_s)) - l(T_key))) / (a_s + eps).
    The poses are constants, the gradient reaches the map through the renders only.
    """
    c2w_key = c2w_key.detach()
    l0 = (render_fn(c2w_key) - target).abs().mean()
    hs = []
    for xi in xis:
        img, ok = warp_by_pose(target, depth, focal, pp, xi)
        m = ok.to(target.dtype)
        a = ((img - target).abs() * m).sum() / (3.0 * m.sum()).clamp_min(1.0)
        c2w = (c2w_key.double() @ delta_pose(xi.detach().to(device=c2w_key.device, dtype=torch.float64))).to(c2w_key.dtype)
        ls = (render_fn(c2w) - target).abs().mean()
        hs.append(F.relu(kappa * a - (ls - l0)) / (a + eps))
    h = torch.stack(hs)
    return h.mean(), float((h.detach() > 0).float().mean())
