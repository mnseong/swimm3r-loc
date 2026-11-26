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


def align_level_torch(stack, X, Jv, Bv, bD, bB, z0, St, fx, fy, cx, cy, scale, transport, huber_k, iters, tol, min_use):
    c3, hs, ws = stack.shape
    C = c3 // 3
    R, t, gh = St[:9].reshape(3, 3).copy(), St[9:12].copy(), St[12:15].copy()
    steps, last, n_use = 0, -1.0, 0
    q = m = use = None
    eye = torch.eye(3, device=X.device)[:C]
    for _ in range(int(iters)):
        Rf = torch.as_tensor(R, dtype=torch.float32, device=X.device)
        tf = torch.as_tensor(t, dtype=torch.float32, device=X.device)
        g = torch.as_tensor(gh[:C], dtype=torch.float32, device=X.device)
        Y = X @ Rf.T + tf
        z = Y[:, 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        u, v = fx * Y[:, 0] / zs + cx, fy * Y[:, 1] / zs + cy
        use = (z > 1e-6) & (u >= 1.0) & (u <= ws - 2.0) & (v >= 1.0) & (v <= hs - 2.0)
        smp = Wp.sample(stack, u[None], v[None])[:, 0]
        qa, qxa, qya = smp[:C], smp[C:2 * C], smp[2 * C:]
        zm = z if transport else z0
        tD, tB = torch.exp(-bD * zm), torch.exp(-bB * zm)
        ma = Jv * tD + Bv * (1.0 - tB)
        dm = (-bD * Jv * tD + bB * Bv * tB) if transport else torch.zeros_like(ma)
        uf = use.float()[None]
        q, m = qa * uf, ma * uf
        r = torch.where(use[None].expand_as(q), g[:, None] * q - m, torch.full_like(q, float("nan")))
        med = r.nanmedian()
        sig = 1.4826 * (r - med).abs().nanmedian() + 1e-6
        n_use = int(use.sum())
        if n_use < int(min_use):
            break
        ru = torch.nan_to_num(r, nan=0.0)
        w = torch.where(ru.abs() > 1e-12, huber_k * sig / ru.abs().clamp_min(1e-12), torch.ones_like(ru)).clamp(max=1.0) * uf
        a0 = g[:, None] * qxa * fx / zs
        a1 = g[:, None] * qya * fy / zs
        a2 = -g[:, None] * (qxa * fx * Y[:, 0] + qya * fy * Y[:, 1]) / (zs * zs)
        pose_rows = [scale * a0, scale * a1, scale * (a2 - dm), Y[:, 1] * a2 - Y[:, 2] * a1 - dm * Y[:, 1],
                     Y[:, 2] * a0 - Y[:, 0] * a2 + dm * Y[:, 0], Y[:, 0] * a1 - Y[:, 1] * a0]
        gain_rows = [eye[:, k][:, None] * q if k < C else torch.zeros_like(q) for k in range(3)]
        rows = torch.stack(pose_rows + gain_rows, -1)
        rw = rows * w[..., None]
        Hm = torch.einsum("cni,cnj->ij", rw.double(), rows.double()).cpu().numpy()
        b = torch.einsum("cni,cn->i", rw.double(), ru.double()).cpu().numpy()
        try:
            xi = np.linalg.solve(Hm + 1e-6 * np.trace(Hm) / 9.0 * np.eye(9), -b)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(xi).all():
            break
        gh[:C] = np.clip(gh[:C] + xi[6:6 + C], 0.25, 4.0)
        Rw = LF.rodrigues(xi[3:6])
        R, t = Rw @ R, Rw @ t + scale * xi[:3]
        steps += 1
        last = float(max(np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:6])))
        if last < tol:
            break
    return np.r_[R.reshape(-1), t, gh], (steps, last, n_use), q, m, use


def align_level(backend, *a):
    if backend == "ext":
        st, info, q, m, use = extension().align_level(*a)
        info = info.numpy()
        return st.numpy(), (int(info[0]), float(info[1]), int(info[2])), q, m, use.bool()
    stack, X, Jv, Bv, bD, bB, z0, St = a[:8]
    return align_level_torch(stack, X, Jv, Bv, bD, bB, z0, St.numpy(), *a[8:])


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
            mk = lambda lo, hi: (lo + (hi - lo) * torch.rand(3, n, generator=g)).to(dev)
            Jv, Bv, bD, bB = mk(0.1, 0.9), mk(0.0, 0.5), mk(0.05, 0.5), mk(0.05, 0.4)
            z0 = X[:, 2].contiguous()
            Rt = torch.tensor(np.r_[LF.rodrigues(np.array([0.01, -0.02, 0.015])).reshape(-1), [0.02, -0.01, 0.03], [0.95, 1.0, 1.04]], dtype=torch.float64)
            args = (stack, X, Jv, Bv, bD, bB, z0, Rt, 30.0, 30.0, (ws - 1) / 2.0, (hs - 1) / 2.0, 2.5, True, 1.345, 4, 1e-9, 16)
            a = align_level("ext", *args)
            b = align_level("torch", *args)
            ok = a[1][0] == b[1][0] and a[1][2] == b[1][2] and float(np.abs(a[0] - b[0]).max()) < 1e-3
            if not ok:
                print("[water] the C++ steps differ from the PyTorch steps on this device (largest %.2e, steps %s against %s), PyTorch is used" % (
                    float(np.abs(a[0] - b[0]).max()), a[1], b[1]), flush=True)
        except Exception as e:
            print("[water] the C++ steps failed on this device (%s: %s), PyTorch is used" % (type(e).__name__, e), flush=True)
            ok = False
    _EXT["ok"][key] = ok
    return ok


def water_align(query, pose, K, parts, levels=(3, 2, 1), iters=(5, 4, 3), transport=True, min_alpha=0.5, tol=1e-4, huber_k=1.345,
                backend="auto", sync_fn=None, render_seconds=0.0):
    dev = query.device
    sync_fn = sync_fn or (lambda: None)
    pose = np.asarray(pose, dtype=np.float64)
    if backend == "auto":
        backend = "ext" if self_check(dev) else "torch"
    sec = dict(render=float(render_seconds), align=0.0)
    out = dict(pose=pose, ok=False, corr=None, valid=0.0, steps=0, stages=sec, seconds=sec["render"], shift=None, backend=backend)
    J, alpha, depth = parts["J"].float(), parts["alpha"].float(), parts["depth"].float()
    lv0 = int(parts.get("level", 0))
    ok = (alpha > min_alpha) & torch.isfinite(depth) & (depth > 1e-6)
    if int(ok.sum()) < 64:
        return out
    sync_fn()
    a = time.time()
    scale = float(depth[ok].median())
    dz = torch.where(ok, depth, torch.zeros_like(depth))
    Rt = start_state()
    steps, last, n_use, hl, q, m, use = 0, None, 0, (1, 1), None, None, None
    with torch.no_grad():
        for s, n_it in zip(levels, iters):
            s = int(s)
            if s < lv0 or min(query.shape[-2:]) // (2 ** s) < 12:
                continue
            fx, fy, cx, cy = Wp.level_intrinsics(K, s)
            Q = Wp.pool_image(query.float(), s)
            stack = gradients(Q)
            Jl = Wp.pool_image(J, s - lv0)
            Bl = Wp.pool_image(parts["binf"].float(), s - lv0)
            d, mk = Wp.pool_depth(dz, ok, s - lv0)
            if Jl.shape[-2:] != Q.shape[-2:]:
                continue
            sel = mk.reshape(-1)
            X = Wp.backproject(d, (fx, fy, cx, cy)).reshape(-1, 3)[sel].contiguous()
            if X.shape[0] < 64:
                continue
            C = Q.shape[0]
            Jv = Jl.reshape(C, -1)[:, sel].contiguous()
            Bv = Bl.reshape(C, -1)[:, sel].contiguous()
            z0 = X[:, 2].contiguous()
            bD, bB = parts["betas"](z0)
            res = align_level(backend, stack, X, Jv, Bv, bD.float().contiguous(), bB.float().contiguous(), z0,
                              torch.as_tensor(Rt, dtype=torch.float64), fx, fy, cx, cy, scale, bool(transport), float(huber_k), int(n_it), float(tol), 64)
            Rt, (st, la, nu), q, m, use = res
            steps += st
            last = la if st else last
            n_use, hl = nu, Q.shape[-2:]
    sync_fn()
    sec["align"] = time.time() - a
    corr = None
    if q is not None and use is not None and int(use.sum()) > 8:
        qa, ma = q[:, use], m[:, use]
        am, qm = ma - ma.mean(), qa - qa.mean()
        corr = float((am * qm).sum() / (am.norm() * qm.norm()).clamp_min(1e-12))
    R, t = Rt[:9].reshape(3, 3), Rt[9:12]
    d4 = np.eye(4)
    d4[:3, :3], d4[:3, 3] = R, t
    new = pose @ np.linalg.inv(d4)
    good = bool(np.isfinite(new).all()) and steps > 0
    ang = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))))
    out.update(pose=new if good else pose, ok=good, corr=corr, valid=float(n_use) / float(hl[0] * hl[1]), steps=steps, last=last, gains=Rt[12:15].tolist(),
               seconds=float(sum(sec.values())), shift=[float(np.linalg.norm(t) / scale), ang])
    return out
