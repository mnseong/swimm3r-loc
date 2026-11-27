import time

import numpy as np
import torch
import torch.nn.functional as F


def quat_matrix(q):
    q = q / q.norm().clamp_min(1e-12)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)]),
        torch.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)]),
        torch.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)])])


def level_intrinsics(K, s):
    k = float(2 ** int(s))
    return float(K[0, 0]) / k, float(K[1, 1]) / k, (float(K[0, 2]) - 0.5 * (k - 1.0)) / k, (float(K[1, 2]) - 0.5 * (k - 1.0)) / k


def pixel_grid(h, w, dev):
    v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    return u, v


def backproject(depth, intr):
    fx, fy, cx, cy = intr
    u, v = pixel_grid(depth.shape[0], depth.shape[1], depth.device)
    return torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)


def project(x, intr):
    fx, fy, cx, cy = intr
    z = x[..., 2]
    zs = torch.where(z.abs() > 1e-9, z, torch.full_like(z, 1e-9))
    return fx * x[..., 0] / zs + cx, fy * x[..., 1] / zs + cy, z


def sample(img, u, v):
    h, w = img.shape[-2:]
    u, v = u.clamp(-2.0, w + 1.0), v.clamp(-2.0, h + 1.0)
    grid =torch.stack([2.0 * u / max(w - 1, 1) - 1.0, 2.0 * v / max(h - 1, 1) - 1.0], -1)[None]
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def pool_image(img, s):
    return img if s == 0 else F.avg_pool2d(img[None], 2 ** s)[0]


def pool_depth(depth, mask, s):
    if s == 0:
        return depth, mask
    m = mask.float()[None, None]
    cover = F.avg_pool2d(m, 2 ** s)
    d = F.avg_pool2d(depth[None, None] * m, 2 ** s) / cover.clamp_min(1e-6)
    return d[0, 0], cover[0, 0] >= 0.5


def nearby_views(pose, n, shift):
    out = []
    for k in range(int(n)):
        a = 2.0 * np.pi * k / float(n)
        m = np.eye(4)
        m[:3, 3] = float(shift) * np.array([np.cos(a), np.sin(a), 0.0])
        out.append(np.asarray(pose, dtype=np.float64) @ m)
    return out


@torch.no_grad()
def visibility_mask(pose, depth, alpha, K, render_depth, views=4, shift=0.02, tau_d=0.01, tau_e=0.3, tau_alpha=0.5):
    ok = (alpha > tau_alpha) & torch.isfinite(depth) & (depth > 1e-6)
    if int(views) <= 0 or not bool(ok.any()):
        return ok, ok
    scale = float(depth[ok].median())
    intr = level_intrinsics(K, 0)
    xa = backproject(torch.where(ok, depth, torch.zeros_like(depth)), intr)
    h, w = depth.shape
    votes = torch.zeros_like(depth)
    for pv in nearby_views(pose, views, shift * scale):
        dv = render_depth(pv)
        rel = torch.as_tensor(np.linalg.inv(pv) @ np.asarray(pose, dtype=np.float64), dtype=torch.float32, device=depth.device)
        u, v, z = project(xa @ rel[:3, :3].T + rel[:3, 3], intr)
        inside = (z > 1e-6) & (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
        good = (torch.isfinite(dv) & (dv > 1e-6)).float()
        seen = sample(torch.stack([torch.where(good > 0.5, dv, torch.zeros_like(dv)), good]), u, v)
        votes += (inside & (seen[1] > 0.999) & ((seen[0] - z).abs() < tau_d * scale)).float()
    return ok & ((1.0 - votes / float(views)) < tau_e), ok


def compose(anchor, q, t):
    d = np.eye(4)
    d[:3, :3] = quat_matrix(torch.as_tensor(np.asarray(q, dtype=np.float64))).numpy()
    d[:3, 3] = np.asarray(t, dtype=np.float64)
    return np.asarray(anchor, dtype=np.float64) @ np.linalg.inv(d)


def refine(query, pose, K, render_rgbd, render_depth, levels=2, steps=20, lr=3e-4, views=4, shift=0.02, tau_d=0.01, tau_e=0.3,
           tau_alpha=0.5, eps_t=0.005, eps_r=0.5, use_mask=True, gain=False, sync=None):
    dev = query.device
    sync = sync or (lambda: None)
    sec = dict(render=0.0, mask=0.0, steps=0.0)
    stat = dict(renders=0, mask_renders=0, rerenders=0, steps=0, mask_share=None, valid=True)
    pose = np.asarray(pose, dtype=np.float64)
    queries = {s: pool_image(query, s) for s in range(int(levels) + 1)}

    def anchor_at(p):
        sync()
        a = time.time()
        with torch.no_grad():
            img, alpha, depth = render_rgbd(p)
            img = img.clamp(0.0, 1.0)
        sync()
        b = time.time()
        if use_mask:
            mask, ok = visibility_mask(p, depth, alpha, K, render_depth, views, shift, tau_d, tau_e, tau_alpha)
            stat["mask_renders"] += int(views)
        else:
            ok = (alpha > 0.05) & torch.isfinite(depth) & (depth > 1e-6)
            mask = ok
        sync()
        c = time.time()
        sec["render"] += b - a
        sec["mask"] += c - b
        stat["renders"] += 1
        scale = float(depth[ok].median()) if bool(ok.any()) else 1.0
        return dict(pose=p, img=img, depth=torch.where(ok, depth, torch.zeros_like(depth)), mask=mask, scale=scale,
                    share=float(mask.float().mean()), any=bool(mask.any()))

    def at_level(an, s):
        d, m = pool_depth(an["depth"], an["mask"], s)
        return pool_image(an["img"], s), backproject(d, level_intrinsics(K, s)), m

    def fresh():
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev, requires_grad=True)
        t = torch.zeros(3, device=dev, requires_grad=True)
        return q, t, torch.optim.Adam([q, t], lr=lr)

    an = anchor_at(pose)
    stat["mask_share"] = an["share"]
    if not an["any"]:
        stat["valid"] = False
        return dict(pose=pose, seconds=float(sum(sec.values())), stages=sec, **stat)
    q, t, opt = fresh()
    for s in range(int(levels), -1, -1):
        intr = level_intrinsics(K, s)
        img_a, xa, m = at_level(an, s)
        hs, ws = img_a.shape[-2:]
        for _ in range(int(steps)):
            sync()
            a = time.time()
            opt.zero_grad(set_to_none=True)
            u, v, z = project(xa @ quat_matrix(q).T + an["scale"] * t, intr)
            use = m & (z > 1e-6) & (u >= 0.0) & (u <= ws - 1.0) & (v >= 0.0) & (v <= hs - 1.0)
            n_use = int(use.sum())
            if n_use < 16:
                break
            seen = sample(queries[s], u, v)
            if gain:
                with torch.no_grad():
                    xa_, xq_ = img_a[:, use], seen[:, use]
                    ma, mq = xa_.mean(1, keepdim=True), xq_.mean(1, keepdim=True)
                    g = ((xa_ - ma) * (xq_ - mq)).sum(1, keepdim=True) / ((xq_ - mq) ** 2).sum(1, keepdim=True).clamp_min(1e-8)
                    g = g.clamp(0.25, 4.0)
                    o = ma - g * mq
                seen = g[:, :, None] * seen + o[:, :, None]
            diff = (img_a - seen) ** 2
            (diff.sum(0)[use].sum() / n_use).backward()
            opt.step()
            stat["steps"] += 1
            with torch.no_grad():
                qn = q / q.norm().clamp_min(1e-12)
                ang = float(torch.rad2deg(2.0 * torch.atan2(qn[1:].norm(), qn[0].abs())))
                moved = float(t.norm())
            sync()
            sec["steps"] += time.time() - a
            if moved > eps_t or ang > eps_r:
                p_new = compose(an["pose"], q.detach().cpu().numpy(), an["scale"] * t.detach().cpu().numpy())
                nxt = anchor_at(p_new)
                if not nxt["any"]:
                    break
                an = nxt
                stat["rerenders"] += 1
                q, t, opt = fresh()
                img_a, xa, m = at_level(an, s)
    out = compose(an["pose"], q.detach().cpu().numpy(), an["scale"] * t.detach().cpu().numpy())
    if not np.isfinite(out).all():
        out, stat["valid"] = pose, False
    return dict(pose=out, seconds=float(sum(sec.values())), stages=sec, **stat)
