import os
import sys
import time

import numpy as np
import torch

import loc_fast as LF
import loc_warp as Wp

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = {"mod": None, "tried": False, "ok": {}}


def extension():
    if not _EXT["tried"]:
        _EXT["tried"] = True
        d = os.path.join(HERE, "csrc")
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            import water_gn_ext
            _EXT["mod"] = water_gn_ext
        except ImportError:
            _EXT["mod"] = None
    return _EXT["mod"]


def gradients(Q):
    gx = torch.zeros_like(Q)
    gy = torch.zeros_like(Q)
    gx[:, :, 1:-1] = 0.5 * (Q[:, :, 2:] - Q[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (Q[:, 2:, :] - Q[:, :-2, :])
    return torch.cat([Q, gx, gy]).contiguous()


def mad_scale(r):
    rf = r.reshape(-1)
    stride = max(1, (rf.numel() + 16383) // 16384)
    rs = rf[::stride]
    med = rs.nanmedian()
    return float(1.4826 * (rs - med).abs().nanmedian() + 1e-6)


def evaluate_torch(stack, X, Iv, Dv, z0, S, fx, fy, cx, cy, transport):
    c3, hs, ws = stack.shape
    C = c3 // 3
    Rf = torch.as_tensor(S[:9].reshape(3, 3), dtype=torch.float32, device=X.device)
    tf = torch.as_tensor(S[9:12], dtype=torch.float32, device=X.device)
    g = torch.as_tensor(S[12:12 + C], dtype=torch.float32, device=X.device)
    Y = X @ Rf.T + tf
    z = Y[:, 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    u, v = fx * Y[:, 0] / zs + cx, fy * Y[:, 1] / zs + cy
    use = (z > 1e-6) & (u >= 1.0) & (u <= ws - 2.0) & (v >= 1.0) & (v <= hs - 2.0)
    smp = Wp.sample(stack, u[None], v[None])[:, 0]
    uf = use.float()[None]
    qa, qxa, qya = smp[:C] * uf, smp[C:2 * C] * uf, smp[2 * C:] * uf
    m = (Iv + Dv * (z - z0)[None]) * uf if transport else Iv * uf
    dm = Dv * uf if transport else torch.zeros_like(m)
    r = torch.where(use[None].expand_as(qa), g[:, None] * qa - m, torch.full_like(qa, float("nan")))
    return dict(Y=Y, zs=zs, use=use, q=qa, qx=qxa, qy=qya, m=m, dm=dm, g=g, r=r, sig=mad_scale(r), n_use=int(use.sum()))


def normal_torch(e, fx, fy, scale, huber_k):
    g, Y, zs, q, qxa, qya, dm = e["g"], e["Y"], e["zs"], e["q"], e["qx"], e["qy"], e["dm"]
    C = q.shape[0]
    uf = e["use"].float()[None]
    ru = torch.nan_to_num(e["r"], nan=0.0)
    w = torch.where(ru.abs() > 1e-12, huber_k * e["sig"] / ru.abs().clamp_min(1e-12), torch.ones_like(ru)).clamp(max=1.0) * uf
    a0 = g[:, None] * qxa * fx / zs
    a1 = g[:, None] * qya * fy / zs
    a2 = -g[:, None] * (qxa * fx * Y[:, 0] + qya * fy * Y[:, 1]) / (zs * zs)
    eye = torch.eye(3, device=q.device)[:C]
    rows = torch.stack([scale * a0, scale * a1, scale * (a2 - dm), Y[:, 1] * a2 - Y[:, 2] * a1 - dm * Y[:, 1],
                        Y[:, 2] * a0 - Y[:, 0] * a2 + dm * Y[:, 0], Y[:, 0] * a1 - Y[:, 1] * a0]
                       + [eye[:, k][:, None] * q if k < C else torch.zeros_like(q) for k in range(3)], -1)
    rw = rows * w[..., None]
    return torch.einsum("cni,cnj->ij", rw.double(), rows.double()).cpu().numpy(), torch.einsum("cni,cn->i", rw.double(), ru.double()).cpu().numpy()


def align_level_torch(stack, X, Iv, Dv, z0, St, fx, fy, cx, cy, scale, transport, huber_k, iters, tol, min_use):
    C = stack.shape[0] // 3
    S = np.array(St, dtype=np.float64)
    best = S.copy()
    steps, last, best_sig, fresh = 0, -1.0, -1.0, False
    e = None
    for _ in range(int(iters)):
        e = evaluate_torch(stack, X, Iv, Dv, z0, S, fx, fy, cx, cy, transport)
        fresh = True
        if e["n_use"] < int(min_use):
            break
        if best_sig >= 0.0 and e["sig"] > best_sig * 1.02:
            S = best.copy()
            steps -= 1
            fresh = False
            break
        best_sig, best = e["sig"], S.copy()
        Hm, b = normal_torch(e, fx, fy, scale, huber_k)
        try:
            xi = np.linalg.solve(Hm + 1e-6 * np.trace(Hm) / 9.0 * np.eye(9), -b)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(xi).all():
            break
        Rw = LF.rodrigues(xi[3:6])
        R, t = S[:9].reshape(3, 3), S[9:12]
        S = np.r_[(Rw @ R).reshape(-1), Rw @ t + scale * xi[:3], S[12:15]]
        S[12:12 + C] = np.clip(S[12:12 + C] + xi[6:6 + C], 0.25, 4.0)
        fresh = False
        steps += 1
        last = float(max(np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:6])))
        if last < tol:
            break
    if not fresh:
        e = evaluate_torch(stack, X, Iv, Dv, z0, S, fx, fy, cx, cy, transport)
        if best_sig >= 0.0 and (e["n_use"] < int(min_use) or e["sig"] > best_sig * 1.02):
            S = best.copy()
            steps -= 1
            e = evaluate_torch(stack, X, Iv, Dv, z0, S, fx, fy, cx, cy, transport)
    steps = max(steps, 0)
    return S, (steps, last, e["n_use"], e["sig"]), e["q"], e["m"], e["use"]


def align_level(backend, *a):
    if backend == "ext":
        st, info, q, m, use = extension().align_level(*a)
        info = info.numpy()
        return st.numpy(), (int(info[0]), float(info[1]), int(info[2]), float(info[3])), q, m, use.bool()
    stack, X, Iv, Dv, z0, St = a[:6]
    return align_level_torch(stack, X, Iv, Dv, z0, St.numpy(), *a[6:])


def start_state():
    return np.r_[np.eye(3).reshape(-1), np.zeros(3), np.ones(3)]


def self_check(dev):
    key = str(dev)
    if key in _EXT["ok"]:
        return _EXT["ok"][key]
    mod = extension()
    ok = False
    if mod is not None and (dev.type != "cuda" or mod.has_cuda()):
        try:
            g = torch.Generator().manual_seed(0)
            hs, ws, n = 40, 56, 900
            img = torch.rand(3, hs, ws, generator=g)
            stack = gradients(LF.box(img, 5)).to(dev)
            X = torch.stack([torch.rand(n, generator=g) * 1.2 - 0.6, torch.rand(n, generator=g) * 0.8 - 0.4, 2.0 + torch.rand(n, generator=g)], -1).to(dev)
            Iv = (0.1 + 0.8 * torch.rand(3, n, generator=g)).to(dev)
            Dv = (-0.2 + 0.25 * torch.rand(3, n, generator=g)).to(dev)
            z0 = X[:, 2].contiguous()
            St = torch.tensor(np.r_[LF.rodrigues(np.array([0.01, -0.02, 0.015])).reshape(-1), [0.02, -0.01, 0.03], [0.95, 1.0, 1.04]], dtype=torch.float64)
            args = (stack, X, Iv, Dv, z0, St, 30.0, 30.0, (ws - 1) / 2.0, (hs - 1) / 2.0, 2.5, True, 1.345, 4, 1e-9, 16)
            a = align_level("ext", *args)
            b = align_level("torch", *args)
            diff = float(np.abs(a[0] - b[0]).max())
            ok = abs(a[1][0] - b[1][0]) <= 1 and abs(a[1][2] - b[1][2]) <= max(2, 0.005 * b[1][2]) and diff < 1e-3
            if not ok:
                print("[water] the C++ steps differ from the PyTorch steps on this device (largest %.2e, steps and points %s against %s), PyTorch is used" % (
                    diff, a[1][:3], b[1][:3]), flush=True)
        except Exception as e:
            print("[water] the C++ steps failed on this device (%s: %s), PyTorch is used" % (type(e).__name__, e), flush=True)
            ok = False
    _EXT["ok"][key] = ok
    return ok


def slopes(Jv, Bv, z0, betas, eps=1e-3):
    bD, bB = [x.float() for x in betas(z0)]
    z1 = z0 * (1.0 + eps)
    bD1, bB1 = [x.float() for x in betas(z1)]
    dz = (z1 - z0).clamp_min(1e-9)[None]
    kD = bD + z0[None] * (bD1 - bD) / dz
    kB = bB + z0[None] * (bB1 - bB) / dz
    return (-kD * Jv * torch.exp(-bD * z0[None]) + kB * Bv * torch.exp(-bB * z0[None])).contiguous()


def water_align(query, pose, K, parts, levels=(3, 2, 1), iters=(5, 4, 3), transport=True, min_alpha=0.5, tol=1e-4, huber_k=1.345,
                backend="auto", sync_fn=None, render_seconds=0.0):
    dev = query.device
    sync_fn = sync_fn or (lambda: None)
    pose = np.asarray(pose, dtype=np.float64)
    if backend == "auto":
        backend = "ext" if self_check(dev) else "torch"
    sec = dict(render=float(render_seconds), align=0.0)
    out = dict(pose=pose, ok=False, corr=None, valid=0.0, steps=0, stages=sec, seconds=sec["render"], shift=None, backend=backend)
    I, J, alpha, depth = parts["I"].float(), parts["J"].float().clamp_min(0.0), parts["alpha"].float(), parts["depth"].float()
    lv0 = int(parts.get("level", 0))
    ok = (alpha > min_alpha) & torch.isfinite(depth) & (depth > 1e-6)
    if int(ok.sum()) < 64:
        return out
    sync_fn()
    a = time.time()
    scale = float(depth[ok].median())
    dz = torch.where(ok, depth, torch.zeros_like(depth))
    St = start_state()
    steps, last, n_use, hl, q, m, use = 0, None, 0, (1, 1), None, None, None
    with torch.no_grad():
        for s, n_it in zip(levels, iters):
            s = int(s)
            if s < lv0 or min(query.shape[-2:]) // (2 ** s) < 12:
                continue
            fx, fy, cx, cy = Wp.level_intrinsics(K, s)
            Q = Wp.pool_image(query.float(), s)
            Il = Wp.pool_image(I, s - lv0)
            if Il.shape[-2:] != Q.shape[-2:]:
                continue
            stack = gradients(Q)
            d, mk = Wp.pool_depth(dz, ok, s - lv0)
            sel = mk.reshape(-1)
            X = Wp.backproject(d, (fx, fy, cx, cy)).reshape(-1, 3)[sel].contiguous()
            if X.shape[0] < 64:
                continue
            C = Q.shape[0]
            Iv = Il.reshape(C, -1)[:, sel].contiguous()
            Jv = Wp.pool_image(J, s - lv0).reshape(C, -1)[:, sel]
            Bv = Wp.pool_image(parts["binf"].float(), s - lv0).reshape(C, -1)[:, sel]
            z0 = X[:, 2].contiguous()
            Dv = slopes(Jv, Bv, z0, parts["betas"])
            res = align_level(backend, stack, X, Iv, Dv, z0, torch.as_tensor(St, dtype=torch.float64), fx, fy, cx, cy, scale, bool(transport),
                              float(huber_k), int(n_it), float(tol), 64)
            St, (st, la, nu, _), q, m, use = res
            steps += st
            last = la if st else last
            n_use, hl = nu, Q.shape[-2:]
    sync_fn()
    sec["align"] = time.time() - a
    corr = None
    if q is not None and use is not None and int(use.sum()) > 8:
        g = torch.as_tensor(St[12:12 + q.shape[0]], dtype=q.dtype, device=q.device)[:, None]
        qa, ma = g * q[:, use], m[:, use]
        am, qm = ma - ma.mean(1, keepdim=True), qa - qa.mean(1, keepdim=True)
        corr = float((am * qm).sum() / (am.norm() * qm.norm()).clamp_min(1e-12))
    R, t = St[:9].reshape(3, 3), St[9:12]
    d4 = np.eye(4)
    d4[:3, :3], d4[:3, 3] = R, t
    new = pose @ np.linalg.inv(d4)
    good = bool(np.isfinite(new).all()) and steps > 0
    ang = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))))
    out.update(pose=new if good else pose, ok=good, corr=corr, valid=float(n_use) / float(hl[0] * hl[1]), steps=steps, last=last, gains=St[12:15].tolist(),
               seconds=float(sum(sec.values())), shift=[float(np.linalg.norm(t) / scale), ang])
    return out
