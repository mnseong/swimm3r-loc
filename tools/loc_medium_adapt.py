import math
import time

import numpy as np
import torch

import loc_fast as LF
import loc_warp as Wp

NAMES = ("veil", "attenuation", "backscatter")


def as_factors(f):
    f = np.ones((3, 3)) if f is None else np.asarray(f, dtype=np.float64).reshape(3, 3)
    return f


def apply_factors(med, f):
    if med is None:
        return None
    f = as_factors(f)
    ref = next((v for v in med.values() if torch.is_tensor(v)), None)
    dev = ref.device if ref is not None else torch.device("cpu")
    fv, fd, fb = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in f]
    out = dict(med)
    if "B_inf_map" in med:
        out["B_inf_map"] = med["B_inf_map"] * fv.to(med["B_inf_map"].dtype)[:, None, None]
    if "B_inf" in med:
        out["B_inf"] = med["B_inf"] * fv.to(med["B_inf"].dtype)
    sc = med["beta_D_scale"] if "beta_D_scale" in med else torch.ones(3, device=dev)
    out["beta_D_scale"] = sc * fd.to(sc.dtype)
    if "params_B" in med:
        pb = med["params_B"].clone()
        pb[0] = pb[0] * fb.to(pb.dtype)
        pb[2] = pb[2] * fb.to(pb.dtype)
        out["params_B"] = pb
    return out


def mad_scale(r):
    rf = r.reshape(-1)
    rs = rf[::max(1, (rf.numel() + 16383) // 16384)]
    return float(1.4826 * (rs - rs.median()).abs().median() + 1e-6)


def huber(r, c):
    a = r.abs()
    return torch.where(a <= c, 0.5 * a * a, c * (a - 0.5 * c))


def coarse_level(parts, query, level, min_side=16):
    lv0 = int(parts.get("level", 0))
    s = max(int(level), lv0)
    while s > lv0 and min(query.shape[-2:]) // (2 ** s) < min_side:
        s -= 1
    return s, lv0


def registration(query, parts, s, min_alpha=0.9):
    lv0 = int(parts.get("level", 0))
    Q = LF.contrast(Wp.pool_image(query.float(), s))
    I = Wp.pool_image(parts["I"].float(), s - lv0)
    if I.shape[-2:] != Q.shape[-2:]:
        return None
    A = LF.contrast(I)
    al = Wp.pool_image(parts["alpha"].float()[None], s - lv0)[0]
    m = (al > min_alpha) & torch.isfinite(Q).all(0) & torch.isfinite(A).all(0)
    if int(m.sum()) < 32:
        return None
    q, a = Q[:, m], A[:, m]
    qm, am = q - q.mean(1, keepdim=True), a - a.mean(1, keepdim=True)
    return float((qm * am).sum() / (qm.norm() * am.norm()).clamp_min(1e-12))


def cell_parity(h, w, size, dev):
    y = torch.div(torch.arange(h, device=dev), int(size), rounding_mode="floor")
    x = torch.div(torch.arange(w, device=dev), int(size), rounding_mode="floor")
    return (y[:, None] + x[None]) % 2


def held_out_mask(alpha, level, cells=0, block=4, min_alpha=0.9):
    h, w = alpha.shape[-2:]
    a = alpha.reshape(h, w)
    return (a > min_alpha) & (cell_parity(h, w, int(block) * 2 ** int(level), a.device) != int(cells))


def masked_psnr(img, ref, mask, min_pixels=16):
    if mask is None or int(mask.sum()) < min_pixels:
        return None
    e = ((img.float().clamp(0.0, 1.0) - ref.float())[:, mask] ** 2).mean()
    return float(-10.0 * torch.log10(e + 1e-12))


def samples(query, parts, level=4, min_alpha=0.9, max_points=4096, cells=None, block=4):
    s, lv0 = coarse_level(parts, query, level)
    Q = Wp.pool_image(query.float(), s)
    I = Wp.pool_image(parts["I"].float(), s - lv0)
    if I.shape[-2:] != Q.shape[-2:]:
        return None
    J = Wp.pool_image(parts["J"].float().clamp_min(0.0), s - lv0)
    B = Wp.pool_image(parts["binf"].float(), s - lv0)
    alpha = parts["alpha"].float()
    dz = parts["depth"].float()
    ok = torch.isfinite(dz) & (dz > 1e-6) & (alpha > min_alpha)
    d, mk = Wp.pool_depth(torch.where(ok, dz, torch.zeros_like(dz)), ok, s - lv0)
    al = Wp.pool_image(alpha[None], s - lv0)[0]
    sel = mk & (al > min_alpha) & torch.isfinite(Q).all(0) & torch.isfinite(I).all(0)
    if cells is not None:
        sel = sel & (cell_parity(sel.shape[0], sel.shape[1], block, sel.device) == int(cells))
    sel = sel.reshape(-1)
    C = Q.shape[0]
    idx = torch.nonzero(sel).reshape(-1)
    if idx.numel() > max_points:
        idx = idx[::int(math.ceil(idx.numel() / float(max_points)))]
    take = lambda x: x.reshape(C, -1)[:, idx].contiguous()
    return dict(q=take(Q), i=take(I), j=take(J), b=take(B), z=d.reshape(-1)[idx].contiguous(), level=s, n=int(idx.numel()),
                share=float(sel.float().mean()))


def water_law(parts, medium=None, law=None):
    if law is not None and medium is not None:
        base = {k: v for k, v in medium.items() if k not in ("B_inf_map", "B_inf")}
        return lambda z, f: [x.float() for x in law(apply_factors(base, f), z)], True
    cache = {}

    def scaled(z, f):
        key = (z.data_ptr(), z.numel())
        if key not in cache:
            cache.clear()
            cache[key] = [x.float() for x in parts["betas"](z)]
        bD, bB = cache[key]
        fd = torch.as_tensor(f[1], dtype=torch.float32, device=z.device)[:, None]
        fb = torch.as_tensor(f[2], dtype=torch.float32, device=z.device)[:, None]
        return fd * bD, fb * bB
    return scaled, False


def per_pixel(sm, betas_of, f):
    bD, bB = betas_of(sm["z"], f)
    z = sm["z"][None]
    fv = torch.as_tensor(f[0], dtype=torch.float32, device=z.device)[:, None]
    return sm["j"] * torch.exp(-bD * z) + fv * sm["b"] * (1.0 - torch.exp(-bB * z))


def adapt_medium(query, parts, medium=None, law=None, start=None, bound=3.0, prior=0.02, min_alpha=0.9, level=4, iters=8, huber_k=1.345, max_points=4096,
                 tol=1e-4, eps=1e-3, min_reg=0.4, reg_level=2, sync_fn=None, cells=None, block=4):
    sync_fn = sync_fn or (lambda: None)
    sync_fn()
    a = time.time()
    lim = math.log(float(bound))
    x0 = np.zeros((3, 3)) if start is None else np.clip(np.log(as_factors(start)), -lim, lim)
    out = dict(ok=False, factors=np.exp(x0).tolist(), log=x0.tolist(), n=0, steps=0, level=coarse_level(parts, query, level)[0], sig_before=None,
               sig_after=None, at_bound=0, law=False, seconds=0.0, reg=None, registered=False, cells=cells)
    with torch.no_grad():
        sr, _ = coarse_level(parts, query, reg_level, min_side=24)
        reg = registration(query, parts, sr, min_alpha) if min_reg is not None else 1.0
        out.update(reg=reg, registered=bool(reg is not None and reg >= float(min_reg or 0.0)))
        sm = samples(query, parts, level=level, min_alpha=min_alpha, max_points=max_points, cells=cells, block=block) if out["registered"] else None
        if sm is None or sm["n"] < 32:
            sync_fn()
            out.update(seconds=time.time() - a, n=0 if sm is None else sm["n"])
            out["apply"] = lambda m, f=np.exp(x0): apply_factors(m, f)
            return out
        betas_of, exact = water_law(parts, medium, law)
        m0 = per_pixel(sm, betas_of, np.ones((3, 3)))
        pred = lambda x: sm["i"] + per_pixel(sm, betas_of, np.exp(x)) - m0
        n = sm["n"]
        r0 = sm["q"] - sm["i"]
        sig0 = mad_scale(r0)
        norm = 1.0 / (n * sig0 * sig0)

        def cost(x, r, c):
            return float(huber(r, c).sum()) * norm + prior * float((x ** 2).sum())

        x = x0.copy()
        p = pred(x)
        r = sm["q"] - p
        c = huber_k * mad_scale(r)
        cur = cost(x, r, c)
        steps = 0
        for _ in range(int(iters)):
            w = (c / r.abs().clamp_min(1e-12)).clamp(max=1.0)
            Jc = []
            for k in range(3):
                e = np.zeros((3, 3))
                e[k] = eps
                Jc.append((pred(x + e) - p) / eps)
            Jm = torch.stack(Jc, -1)
            H = torch.einsum("cnk,cnl->ckl", Jm * w[..., None], Jm).double().cpu().numpy() * norm
            g = torch.einsum("cnk,cn->ck", Jm * w[..., None], r).double().cpu().numpy() * norm
            step = np.zeros((3, 3))
            for ch in range(3):
                Hc = H[ch] + 2.0 * prior * np.eye(3)
                gc = g[ch] - 2.0 * prior * x[:, ch]
                try:
                    step[:, ch] = np.linalg.solve(Hc + 1e-9 * np.eye(3), gc)
                except np.linalg.LinAlgError:
                    step[:, ch] = 0.0
            if not np.isfinite(step).all():
                break
            acc = False
            for _ in range(5):
                xn = np.clip(x + step, -lim, lim)
                pn = pred(xn)
                rn = sm["q"] - pn
                cn = cost(xn, rn, c)
                if cn <= cur:
                    acc = True
                    break
                step = 0.5 * step
            if not acc:
                break
            moved = float(np.abs(xn - x).max())
            x, p, r, cur = xn, pn, rn, cn
            steps += 1
            c = huber_k * mad_scale(r)
            cur = cost(x, r, c)
            if moved < tol:
                break
        sig1 = mad_scale(r)
    sync_fn()
    f = np.exp(x)
    out.update(ok=True, factors=f.tolist(), log=x.tolist(), n=n, steps=steps, level=sm["level"], share=sm["share"], sig_before=sig0, sig_after=sig1,
               at_bound=int((np.abs(x) >= lim - 1e-6).sum()), law=bool(exact), seconds=time.time() - a)
    out["apply"] = lambda m, f=f: apply_factors(m, f)
    return out


def fit_gain(query, parts, bound=3.0, min_alpha=0.9, level=4, iters=6, huber_k=1.345, max_points=4096, min_reg=0.4, reg_level=2, sync_fn=None, cells=None,
             block=4, **_):
    sync_fn = sync_fn or (lambda: None)
    sync_fn()
    a = time.time()
    out = dict(ok=False, gain=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], n=0, level=coarse_level(parts, query, level)[0], sig_before=None, sig_after=None,
               seconds=0.0, reg=None, registered=False, cells=cells)
    with torch.no_grad():
        sr, _ = coarse_level(parts, query, reg_level, min_side=24)
        reg = registration(query, parts, sr, min_alpha) if min_reg is not None else 1.0
        out.update(reg=reg, registered=bool(reg is not None and reg >= float(min_reg or 0.0)))
        sm = samples(query, parts, level=level, min_alpha=min_alpha, max_points=max_points, cells=cells, block=block) if out["registered"] else None
        if sm is None or sm["n"] < 32:
            sync_fn()
            out.update(seconds=time.time() - a, n=0 if sm is None else sm["n"])
            return out
        q, i = sm["q"], sm["i"]
        g = torch.ones(q.shape[0], 1, device=q.device)
        o = torch.zeros_like(g)
        sig0 = mad_scale(q - i)
        lo, hi = 1.0 / float(bound), float(bound)
        for _ in range(int(iters)):
            r = q - (g * i + o)
            c = huber_k * mad_scale(r)
            w = (c / r.abs().clamp_min(1e-12)).clamp(max=1.0)
            sw = w.sum(1, keepdim=True).clamp_min(1e-12)
            mi, mq = (w * i).sum(1, keepdim=True) / sw, (w * q).sum(1, keepdim=True) / sw
            g = ((w * (i - mi) * (q - mq)).sum(1, keepdim=True) / (w * (i - mi) ** 2).sum(1, keepdim=True).clamp_min(1e-12)).clamp(lo, hi)
            o = mq - g * mi
        sig1 = mad_scale(q - (g * i + o))
        gl, ol = g.reshape(-1).double().cpu().numpy(), o.reshape(-1).double().cpu().numpy()
    sync_fn()
    if not (np.isfinite(gl).all() and np.isfinite(ol).all()):
        out.update(seconds=time.time() - a, n=sm["n"])
        return out
    out.update(ok=True, gain=gl.tolist(), offset=ol.tolist(), n=sm["n"], level=sm["level"], share=sm["share"], sig_before=sig0, sig_after=sig1,
               seconds=time.time() - a)
    return out


def apply_gain(img, fit):
    if fit is None or not fit.get("ok"):
        return img
    g = torch.as_tensor(fit["gain"], dtype=img.dtype, device=img.device)[:, None, None]
    o = torch.as_tensor(fit["offset"], dtype=img.dtype, device=img.device)[:, None, None]
    return (g * img + o).clamp(0.0, 1.0)


def adapted_parts(parts, fit, medium=None, law=None, I=None):
    f = as_factors(fit["factors"])
    betas_of, _ = water_law(parts, medium, law)
    out = dict(parts, binf=parts["binf"] * torch.as_tensor(f[0], dtype=parts["binf"].dtype, device=parts["binf"].device)[:, None, None],
               betas=lambda z, f=f: tuple(betas_of(z, f)))
    if I is not None:
        out["I"] = I
    return out


def summary(fits):
    f = np.array([as_factors(x["factors"]) for x in fits if x.get("ok")])
    if not len(f):
        return None
    out = dict(n=int(len(f)), at_bound=int(sum(x.get("at_bound", 0) for x in fits if x.get("ok"))))
    for k, name in enumerate(NAMES):
        out[name] = dict(median=np.median(f[:, k], 0).tolist(), p25=np.percentile(f[:, k], 25, 0).tolist(), p75=np.percentile(f[:, k], 75, 0).tolist(),
                         min=f[:, k].min(0).tolist(), max=f[:, k].max(0).tolist())
    sb = [x["sig_before"] for x in fits if x.get("ok") and x.get("sig_before") is not None]
    sa = [x["sig_after"] for x in fits if x.get("ok") and x.get("sig_after") is not None]
    if sb and sa:
        out.update(sig_before=float(np.median(sb)), sig_after=float(np.median(sa)))
    return out
