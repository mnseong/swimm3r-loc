import time

import numpy as np
import torch
import torch.nn.functional as F

import loc_geom as G
import loc_warp as Wp


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def token_grid(loc):
    t = int(loc.tokens.shape[1])
    ps = int(round(float(np.sqrt(loc.h * loc.w / float(t)))))
    gh, gw = loc.h // ps, loc.w // ps
    if gh * gw != t:
        raise ValueError("token grid %d x %d does not hold %d tokens" % (gh, gw, t))
    return ps, gh, gw


def token_points(loc, i, ps):
    a, b = ps // 2 - 1, ps // 2
    x, v = loc.xyz[i], loc.valid[i]
    p = 0.25 * (x[a::ps, a::ps] + x[a::ps, b::ps] + x[b::ps, a::ps] + x[b::ps, b::ps])
    ok = v[a::ps, a::ps] & v[a::ps, b::ps] & v[b::ps, a::ps] & v[b::ps, b::ps]
    return p.reshape(-1, 3), ok.reshape(-1)


@torch.no_grad()
def token_matches(loc, fq, idx, temp=0.07):
    ps, gh, gw = token_grid(loc)
    q = F.normalize(fq[0].float() - loc.mean_token, dim=-1)
    p2d, p3d, n_mutual = [], [], 0
    for i in idx:
        m = F.normalize(loc.tokens[i].float() - loc.mean_token, dim=-1)
        sim = m @ q.t()
        best_q = sim.argmax(1)
        best_m = sim.argmax(0)
        ids = torch.arange(sim.shape[0], device=sim.device)
        mutual = best_m[best_q] == ids
        pts, ok = token_points(loc, i, ps)
        keep = mutual & ok
        n_mutual += int(mutual.sum())
        if not bool(keep.any()):
            continue
        mi, qi = ids[keep], best_q[keep]
        qy, qx = torch.div(qi, gw, rounding_mode="floor"), qi % gw
        row = F.pad(sim[mi].reshape(-1, 1, gh, gw), (1, 1, 1, 1), value=-1e4)
        win = torch.stack([row[torch.arange(len(mi), device=sim.device), 0, qy + dy, qx + dx] for dy in range(3) for dx in range(3)], 1)
        w = torch.softmax(win / float(temp), dim=1)
        off = torch.tensor([[dx - 1.0, dy - 1.0] for dy in range(3) for dx in range(3)], device=sim.device)
        d = w @ off
        px = torch.stack([(qx.float() + 0.5 + d[:, 0]) * ps - 0.5, (qy.float() + 0.5 + d[:, 1]) * ps - 0.5], 1)
        p2d.append(px.double().cpu().numpy())
        p3d.append(pts[mi].double().cpu().numpy())
    if not p2d:
        return np.zeros((0, 2)), np.zeros((0, 3)), n_mutual
    return np.concatenate(p2d), np.concatenate(p3d), n_mutual


class TokenAnchors:
    def __init__(self, loc, cell=1.0, min_views=1):
        ps, gh, gw = token_grid(loc)
        self.ps, self.gh, self.gw = ps, gh, gw
        feats, pts, frames = [], [], []
        with torch.no_grad():
            for i in range(loc.n):
                p, ok = token_points(loc, i, ps)
                f = F.normalize(loc.tokens[i].float() - loc.mean_token, dim=-1)
                feats.append(f[ok])
                pts.append(p[ok])
                frames.append(torch.full((int(ok.sum()),), i, device=p.device, dtype=torch.long))
            feats, pts, frames = torch.cat(feats), torch.cat(pts), torch.cat(frames)
            self.n_tokens = int(pts.shape[0])
            depth = ((pts - loc.centres[frames]) * loc.forward[frames]).sum(-1)
            self.voxel = float(cell) * ps * float(depth.clamp_min(1e-6).median()) / loc.focal
            key = torch.floor((pts - pts.min(0).values) / self.voxel).long()
            dims = key.max(0).values + 1
            flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
            uniq, inv = torch.unique(flat, return_inverse=True)
            m = int(uniq.numel())
            cnt = torch.zeros(m, device=pts.device).index_add_(0, inv, torch.ones(len(inv), device=pts.device))
            xyz = torch.zeros(m, 3, device=pts.device).index_add_(0, inv, pts) / cnt[:, None]
            feat = F.normalize(torch.zeros(m, feats.shape[1], device=pts.device).index_add_(0, inv, feats), dim=-1)
            pairs = torch.unique(inv * loc.n + frames)
            views = torch.zeros(m, device=pts.device).index_add_(0, torch.div(pairs, loc.n, rounding_mode="floor"), torch.ones(len(pairs), device=pts.device))
            keep = views >= float(min_views)
            if int(keep.sum()) < 64:
                keep = torch.ones_like(keep)
            self.xyz, self.feat, self.views = xyz[keep], feat[keep], views[keep]
        self.m = int(self.xyz.shape[0])

    def megabytes(self):
        return (self.feat.numel() + self.xyz.numel()) * 4 / 1e6

    @torch.no_grad()
    def match(self, loc, fq, temp=0.07):
        q = F.normalize(fq[0].float() - loc.mean_token, dim=-1)
        sim = q @ self.feat.t()
        best_a = sim.argmax(1)
        best_q = sim.argmax(0)
        t = torch.arange(sim.shape[0], device=sim.device)
        mutual = best_q[best_a] == t
        if not bool(mutual.any()):
            return np.zeros((0, 2)), np.zeros((0, 3)), 0
        qi, ai = t[mutual], best_a[mutual]
        gh, gw, ps = self.gh, self.gw, self.ps
        qy, qx = torch.div(qi, gw, rounding_mode="floor"), qi % gw
        row = F.pad(sim[:, ai].t().reshape(-1, 1, gh, gw), (1, 1, 1, 1), value=-1e4)
        ar = torch.arange(len(qi), device=sim.device)
        win = torch.stack([row[ar, 0, qy + dy, qx + dx] for dy in range(3) for dx in range(3)], 1)
        w = torch.softmax(win / float(temp), dim=1)
        off = torch.tensor([[dx - 1.0, dy - 1.0] for dy in range(3) for dx in range(3)], device=sim.device)
        d = w @ off
        px = torch.stack([(qx.float() + 0.5 + d[:, 0]) * ps - 0.5, (qy.float() + 0.5 + d[:, 1]) * ps - 0.5], 1)
        return px.double().cpu().numpy(), self.xyz[ai].double().cpu().numpy(), int(mutual.sum())


def box(x, k):
    return F.avg_pool2d(F.pad(x[None], (k // 2,) * 4, mode="replicate"), k, stride=1)[0]


def contrast(img, k=7, eps=0.02):
    lum = img.mean(0, keepdim=True)
    mu = box(lum, k)
    var = (box(lum * lum, k) - mu * mu).clamp_min(0.0)
    return (lum - mu) / torch.sqrt(var + eps * eps)


def rodrigues(w):
    th = float(np.linalg.norm(w))
    k = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + k
    return np.eye(3) + np.sin(th) / th * k + (1.0 - np.cos(th)) / (th * th) * k @ k


def direct_align(query, pose, K, render_rgbd, levels=(3, 2, 1), iters=(5, 4, 3), photo="lcn", min_alpha=0.5, tol=1e-4, sync_fn=None):
    dev = query.device
    sync_fn = sync_fn or (lambda: None)
    pose = np.asarray(pose, dtype=np.float64)
    sec = dict(render=0.0, align=0.0)
    sync_fn()
    a = time.time()
    with torch.no_grad():
        img, alpha, depth = render_rgbd(pose)
        img = img.clamp(0.0, 1.0)
    sync_fn()
    sec["render"] = time.time() - a
    out = dict(pose=pose, ok=False, corr=None, valid=0.0, steps=0, stages=sec, seconds=sec["render"], shift=None)
    ok = (alpha > min_alpha) & torch.isfinite(depth) & (depth > 1e-6)
    if int(ok.sum()) < 64:
        return out
    a = time.time()
    scale = float(depth[ok].median())
    dz = torch.where(ok, depth, torch.zeros_like(depth))
    R, t = np.eye(3), np.zeros(3)
    corr, valid, steps, last = None, 0.0, 0, None
    with torch.no_grad():
        for s, n_it in zip(levels, iters):
            s = int(s)
            if min(query.shape[-2:]) // (2 ** s) < 12:
                continue
            fx, fy, cx, cy = Wp.level_intrinsics(K, s)
            A, Q = Wp.pool_image(img, s), Wp.pool_image(query, s)
            d, m = Wp.pool_depth(dz, ok, s)
            if photo == "lcn":
                A, Q = contrast(A), contrast(Q)
            gx = torch.zeros_like(Q)
            gy = torch.zeros_like(Q)
            gx[:, :, 1:-1] = 0.5 * (Q[:, :, 2:] - Q[:, :, :-2])
            gy[:, 1:-1, :] = 0.5 * (Q[:, 2:, :] - Q[:, :-2, :])
            stack = torch.cat([Q, gx, gy])
            c = Q.shape[0]
            hs, ws = Q.shape[-2:]
            X = Wp.backproject(d, (fx, fy, cx, cy)).reshape(-1, 3)[m.reshape(-1)]
            Av = A.reshape(c, -1)[:, m.reshape(-1)]
            for _ in range(int(n_it)):
                Rt = torch.as_tensor(R, dtype=torch.float32, device=dev)
                tt = torch.as_tensor(t, dtype=torch.float32, device=dev)
                Y = X @ Rt.T + tt
                z = Y[:, 2]
                zs = torch.where(z > 1e-6, z, torch.ones_like(z))
                u, v = fx * Y[:, 0] / zs + cx, fy * Y[:, 1] / zs + cy
                use = (z > 1e-6) & (u >= 1.0) & (u <= ws - 2.0) & (v >= 1.0) & (v <= hs - 2.0)
                n_use = int(use.sum())
                if n_use < 64:
                    break
                smp = Wp.sample(stack, u[use][None], v[use][None])[:, 0]
                q, qx, qy = smp[:c], smp[c:2 * c], smp[2 * c:]
                av = Av[:, use]
                if photo != "lcn":
                    mq, ma = q.mean(1, keepdim=True), av.mean(1, keepdim=True)
                    g = (((av - ma) * (q - mq)).sum(1, keepdim=True) / ((q - mq) ** 2).sum(1, keepdim=True).clamp_min(1e-8)).clamp(0.25, 4.0)
                    q, qx, qy = g * q + (ma - g * mq), g * qx, g * qy
                r = q - av
                sig = 1.4826 * float((r - r.median()).abs().median()) + 1e-6
                wgt = (1.345 * sig / r.abs().clamp_min(1e-12)).clamp(max=1.0)
                Yu, zu = Y[use], zs[use]
                jx = torch.stack([fx / zu, torch.zeros_like(zu), -fx * Yu[:, 0] / (zu * zu)], 1)
                jy = torch.stack([torch.zeros_like(zu), fy / zu, -fy * Yu[:, 1] / (zu * zu)], 1)
                av3 = qx[:, :, None] * jx[None] + qy[:, :, None] * jy[None]
                J = torch.cat([scale * av3, torch.cross(Yu[None].expand_as(av3), av3, dim=-1)], -1)
                Jw = J * wgt[:, :, None]
                Hm = torch.einsum("cni,cnj->ij", Jw, J).double().cpu().numpy()
                b = torch.einsum("cni,cn->i", Jw, r).double().cpu().numpy()
                try:
                    xi = np.linalg.solve(Hm + 1e-6 * np.trace(Hm) / 6.0 * np.eye(6), -b)
                except np.linalg.LinAlgError:
                    break
                if not np.isfinite(xi).all():
                    break
                Rw = rodrigues(xi[3:])
                R, t = Rw @ R, Rw @ t + scale * xi[:3]
                steps += 1
                last = float(max(np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:])))
                am, qm = av - av.mean(), q - q.mean()
                corr = float((am * qm).sum() / (am.norm() * qm.norm()).clamp_min(1e-12))
                valid = n_use / float(hs * ws)
                if last < tol:
                    break
    sync_fn()
    sec["align"] = time.time() - a
    d4 = np.eye(4)
    d4[:3, :3], d4[:3, 3] = R, t
    new = pose @ np.linalg.inv(d4)
    good = bool(np.isfinite(new).all()) and steps > 0
    ang = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))))
    out.update(pose=new if good else pose, ok=good, corr=corr, valid=float(valid), steps=steps, last=last, seconds=float(sum(sec.values())),
               shift=[float(np.linalg.norm(t) / scale), ang])
    return out


def localize_fast(loc, img, render_at, topk=2, thr_px=None, half=False, photo="lcn", levels=(3, 2, 1), iters=(5, 4, 3), anchors=None):
    t = {}
    sync(loc.dev)
    a = time.time()
    fq = loc.encode(img, half=half)
    sync(loc.dev)
    t["encoder"] = time.time() - a
    a = time.time()
    ps = token_grid(loc)[0]
    thr = float(thr_px) if thr_px else 0.5 * ps
    if anchors is None:
        idx, _ = loc.retrieve(fq, topk)
        p2d, p3d, n_mutual = token_matches(loc, fq, idx)
        start = loc.poses[idx[0]]
    else:
        p2d, p3d, n_mutual = anchors.match(loc, fq)
        idx, start = None, None
    sync(loc.dev)
    t["tokens"] = time.time() - a
    a = time.time()
    sol = G.solve_pnp(p3d, p2d, loc.K, thr, start) if len(p2d) >= 6 else None
    coarse_ok = sol is not None and bool(np.isfinite(sol["c2w"]).all())
    if idx is None:
        idx = loc.near(sol["c2w"], 1) if coarse_ok else loc.retrieve(fq, 1)[0]
        start = loc.poses[idx[0]]
    t["pnp"] = time.time() - a
    coarse = np.asarray(sol["c2w"], dtype=np.float64) if coarse_ok else np.asarray(start, dtype=np.float64)
    inl = int(sol["inliers"]) if coarse_ok else 0
    al = direct_align(img, coarse, loc.K, lambda p: render_at(p, idx[0]), levels=levels, iters=iters, photo=photo, sync_fn=lambda: sync(loc.dev))
    t["render"], t["align"] = al["stages"]["render"], al["stages"]["align"]
    return dict(ok=bool(coarse_ok and al["ok"]), pose=al["pose"], coarse=coarse, idx=list(idx), matches=int(len(p2d)), mutual=int(n_mutual), inliers=inl,
                corr=al["corr"], valid=al["valid"], steps=al["steps"], shift=al["shift"], stages=t, seconds=float(sum(t.values())), fq=fq)


def passes(r, min_inliers=12, min_ratio=0.2, min_corr=0.25, min_valid=0.25, max_shift=(0.5, 20.0)):
    if not r["ok"] or r["corr"] is None or r["shift"] is None:
        return False
    return bool(r["inliers"] >= max(min_inliers, min_ratio * r["matches"]) and r["corr"] >= min_corr and r["valid"] >= min_valid
                and r["shift"][0] <= max_shift[0] and r["shift"][1] <= max_shift[1])


def localize_tiered(loc, img, render_at, check=None, lean=True, half=False, **kw):
    r = localize_fast(loc, img, render_at, half=half, **kw)
    fq = r.pop("fq")
    r["fallback"] = False
    if passes(r, **(check or {})):
        return r
    b = loc.localize(img, idx=r["idx"][:1], lean=lean, half=half, fq=fq)
    stages = dict(r["stages"])
    for k, v in b["stages"].items():
        stages["fallback_" + k] = v
    return dict(r, ok=bool(b["ok"]), pose=b["pose"], fallback=True, stages=stages, seconds=r["seconds"] + b["seconds"], full_inliers=b["inliers"],
                full_matches=b["matches"])
