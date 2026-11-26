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
    def __init__(self, loc, cell=0.25, min_views=1, dim=0):
        t0 = time.time()
        ps, gh, gw = token_grid(loc)
        self.ps, self.gh, self.gw = ps, gh, gw
        dev = loc.tokens.device
        with torch.no_grad():
            pts, oks, frames = [], [], []
            for i in range(loc.n):
                p, ok = token_points(loc, i, ps)
                pts.append(p[ok].float())
                oks.append(ok)
                frames.append(torch.full((int(ok.sum()),), i, device=dev, dtype=torch.long))
            pts, frames = torch.cat(pts), torch.cat(frames)
            self.n_tokens = int(pts.shape[0])
            depth = ((pts - loc.centres[frames]) * loc.forward[frames]).sum(-1)
            self.voxel = float(cell) * ps * float(depth.clamp_min(1e-6).median()) / loc.focal
            key = torch.floor((pts - pts.min(0).values) / self.voxel).long()
            dims = key.max(0).values + 1
            flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
            uniq, inv = torch.unique(flat, return_inverse=True)
            m = int(uniq.numel())
            one = torch.ones(len(inv), device=dev)
            cnt = torch.zeros(m, device=dev).index_add_(0, inv, one)
            xyz = torch.zeros(m, 3, device=dev).index_add_(0, inv, pts) / cnt[:, None]
            pairs = torch.unique(inv * loc.n + frames)
            views = torch.zeros(m, device=dev).index_add_(0, torch.div(pairs, loc.n, rounding_mode="floor"), torch.ones(len(pairs), device=dev))
            feat = torch.zeros(m, int(loc.tokens.shape[-1]), device=dev)
            at = 0
            for i in range(loc.n):
                k = int(oks[i].sum())
                if k:
                    feat.index_add_(0, inv[at:at + k], F.normalize(loc.tokens[i].float() - loc.mean_token, dim=-1)[oks[i]])
                at += k
            self.all = m
            keep = views >= float(min_views)
            self.min_views = int(min_views)
            if int(keep.sum()) < 64:
                keep, self.min_views = torch.ones_like(keep), 1
            xyz, feat, views = xyz[keep], F.normalize(feat[keep], dim=-1), views[keep]
            self.basis, self.kept = None, 1.0
            if dim and int(dim) < feat.shape[1]:
                cov = (feat.t() @ feat).double().cpu()
                w, v = torch.linalg.eigh(cov)
                self.basis = v[:, -int(dim):].float().to(dev)
                self.kept = float(w[-int(dim):].sum() / w.sum().clamp_min(1e-12))
                feat = F.normalize(feat @ self.basis, dim=-1)
            self.xyz, self.feat, self.views = xyz, feat, views
        self.m, self.dim, self.cell = int(self.xyz.shape[0]), int(self.feat.shape[1]), float(cell)
        self.seconds = time.time() - t0

    def megabytes(self):
        n = self.feat.numel() + self.xyz.numel() + (self.basis.numel() if self.basis is not None else 0)
        return n * 4 / 1e6

    def subtoken(self, sim, qi, ai, temp):
        gh, gw, ps = self.gh, self.gw, self.ps
        qy, qx = torch.div(qi, gw, rounding_mode="floor"), qi % gw
        row = F.pad(sim[:, ai].t().reshape(-1, 1, gh, gw), (1, 1, 1, 1), value=-1e4)
        ar = torch.arange(len(qi), device=sim.device)
        win = torch.stack([row[ar, 0, qy + dy, qx + dx] for dy in range(3) for dx in range(3)], 1)
        w = torch.softmax(win / float(temp), dim=1)
        off = torch.tensor([[dx - 1.0, dy - 1.0] for dy in range(3) for dx in range(3)], device=sim.device)
        d = w @ off
        return torch.stack([(qx.float() + 0.5 + d[:, 0]) * ps - 0.5, (qy.float() + 0.5 + d[:, 1]) * ps - 0.5], 1)

    @torch.no_grad()
    def match(self, loc, fq, temp=0.07):
        q = F.normalize(fq[0].float() - loc.mean_token, dim=-1)
        if self.basis is not None:
            q = F.normalize(q @ self.basis, dim=-1)
        sim = q @ self.feat.t()
        best_a = sim.argmax(1)
        best_q = sim.argmax(0)
        t = torch.arange(sim.shape[0], device=sim.device)
        mutual = best_q[best_a] == t
        state = dict(sim=sim, best_q=best_q, temp=temp)
        if not bool(mutual.any()):
            return np.zeros((0, 2)), np.zeros((0, 3)), 0, state
        qi, ai = t[mutual], best_a[mutual]
        px = self.subtoken(sim, qi, ai, temp)
        return px.double().cpu().numpy(), self.xyz[ai].double().cpu().numpy(), int(mutual.sum()), state

    @torch.no_grad()
    def guided(self, state, pose, K, reach=1):
        sim, best_q = state["sim"], state["best_q"]
        c = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=sim.device)
        pc = (self.xyz - c[:3, 3]) @ c[:3, :3]
        z = pc[:, 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        tx = torch.floor((float(K[0, 0]) * pc[:, 0] / zs + float(K[0, 2]) + 0.5) / self.ps)
        ty = torch.floor((float(K[1, 1]) * pc[:, 1] / zs + float(K[1, 2]) + 0.5) / self.ps)
        by, bx = torch.div(best_q, self.gw, rounding_mode="floor").float(), (best_q % self.gw).float()
        seen = (z > 1e-6) & (tx >= 0) & (tx < self.gw) & (ty >= 0) & (ty < self.gh)
        ok = seen & ((bx - tx).abs() <= reach) & ((by - ty).abs() <= reach)
        ai = torch.nonzero(ok)[:, 0]
        if len(ai) == 0:
            return np.zeros((0, 2)), np.zeros((0, 3)), int(seen.sum())
        px = self.subtoken(sim, best_q[ai], ai, state["temp"])
        return px.double().cpu().numpy(), self.xyz[ai].double().cpu().numpy(), int(seen.sum())


@torch.no_grad()
def vote_frame(loc, p3d):
    if len(p3d) == 0:
        return None
    P = torch.as_tensor(np.asarray(p3d), dtype=torch.float32, device=loc.dev)
    pc = torch.einsum("nkj,nji->nki", P[None] - loc.centres[:, None], loc.c2w[:, :3, :3])
    z = pc[..., 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    u = float(loc.K[0, 0]) * pc[..., 0] / zs + float(loc.K[0, 2])
    v = float(loc.K[1, 1]) * pc[..., 1] / zs + float(loc.K[1, 2])
    inside = (z > 1e-6) & (u >= 0) & (u <= loc.w - 1) & (v >= 0) & (v <= loc.h - 1)
    off = ((u - 0.5 * (loc.w - 1)).abs() / loc.w + (v - 0.5 * (loc.h - 1)).abs() / loc.h) * inside
    score = inside.sum(1).float() - off.sum(1) / inside.sum(1).clamp_min(1)
    return int(score.argmax())


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


def localize_fast(loc, img, render_at, topk=2, thr_px=None, half=False, photo="lcn", levels=(3, 2, 1), iters=(5, 4, 3), anchors=None, guided=True, support=64,
                  water=None, transport=True, backend="auto"):
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
        p2d, p3d, n_mutual, state = anchors.match(loc, fq)
        vote = vote_frame(loc, p3d)
        idx = None
        start = loc.poses[vote] if vote is not None else None
    sync(loc.dev)
    t["tokens"] = time.time() - a
    a = time.time()
    sol = G.solve_pnp(p3d, p2d, loc.K, thr, start) if len(p2d) >= 6 else None
    coarse_ok = sol is not None and bool(np.isfinite(sol["c2w"]).all())
    t["pnp"] = time.time() - a
    n_first, n_seen, n_agree, densified = int(len(p2d)), 0, 0, False
    if anchors is not None and coarse_ok and guided:
        a = time.time()
        g2d, g3d, n_seen = anchors.guided(state, sol["c2w"], loc.K)
        n_agree = int(len(g2d))
        sync(loc.dev)
        t["tokens"] += time.time() - a
        a = time.time()
        if int(sol["inliers"]) < int(support) and len(g2d) > len(p2d):
            x = G.polish(G._to_x(sol["c2w"]), g3d, g2d, loc.K, thr)
            if x is not None:
                p2d, p3d, densified = g2d, g3d, True
                sol = dict(c2w=G._to_c2w(x), inliers=int((G._errors(x, g3d, g2d, loc.K) < thr).sum()))
        t["pnp"] += time.time() - a
    if idx is None:
        a = time.time()
        idx = loc.near(sol["c2w"], 1) if coarse_ok else ([vote] if vote is not None else loc.retrieve(fq, 1)[0])
        start = loc.poses[idx[0]]
        t["pnp"] += time.time() - a
    coarse = np.asarray(sol["c2w"], dtype=np.float64) if coarse_ok else np.asarray(start, dtype=np.float64)
    inl = int(sol["inliers"]) if coarse_ok else 0
    if water is not None:
        import loc_water as LW
        sync(loc.dev)
        a = time.time()
        with torch.no_grad():
            parts = water(coarse, idx[0])
        sync(loc.dev)
        al = LW.water_align(img, coarse, loc.K, parts, levels=levels, iters=iters, transport=transport, backend=backend, sync_fn=lambda: sync(loc.dev),
                            render_seconds=time.time() - a)
    else:
        al = direct_align(img, coarse, loc.K, lambda p: render_at(p, idx[0]), levels=levels, iters=iters, photo=photo, sync_fn=lambda: sync(loc.dev))
    t["render"], t["align"] = al["stages"]["render"], al["stages"]["align"]
    return dict(ok=bool(coarse_ok and al["ok"]), pose=al["pose"], coarse=coarse, idx=list(idx), matches=int(len(p2d)), mutual=int(n_mutual), inliers=inl,
                first=n_first, seen=int(n_seen), agree=int(n_agree), densified=bool(densified), backend=al.get("backend", "torch"),
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
