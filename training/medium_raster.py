import torch

from training.physics_splat import background, beta_at


SGA_STAGE2 = dict(grad_deconf_enabled=True, grad_deconf_gamma=1.0, grad_deconf_lambda_med=1.0, grad_deconf_eps=1e-12, grad_deconf_residual_eta=0.15,
                  grad_deconf_fade=0.0, grad_deconf_apply_means2d=True, grad_deconf_apply_conic=False, grad_deconf_rho_mode=0,
                  grad_deconf_apply_opacity=False, grad_deconf_gamma_opacity=0.0, grad_deconf_eta_opacity=0.0, grad_deconf_gamma_medium=0.0,
                  grad_deconf_lambda_obj=1.0, grad_deconf_medium_bs_floor=0.3, grad_deconf_medium_attn_exact=True)


def scale_grad(x, s):
    return x.detach() + s * (x - x.detach())


def expected_depth(depth_acc, alpha, unit=1.0, min_alpha=0.05):
    a = alpha.reshape(alpha.shape[-2:])
    d = depth_acc.reshape(a.shape) / a.clamp_min(1e-6) / float(unit)
    ok = (a > min_alpha) & torch.isfinite(d) & (d > 1e-6)
    fill = d[ok].median() if bool(ok.any()) else d.new_tensor(1.0)
    return torch.where(ok, d, fill)


def medium_maps(phys, med1, z, unit=1.0):
    h, w = z.shape[-2:]
    zz = z.reshape(-1)
    ok = torch.isfinite(zz) & (zz > 1e-6)
    fill = zz[ok].median() if bool(ok.any()) else zz.new_tensor(1.0)
    bD, bB = beta_at(phys, med1, torch.where(ok, zz, fill))
    rgb = background(med1, h, w).permute(1, 2, 0)
    u = float(unit)
    return rgb.float().contiguous(), (bB.t().reshape(h, w, 3) / u).float().contiguous(), (bD.t().reshape(h, w, 3) / u).float().contiguous()


def render_medium_view(means, opacity, scales, quats, colors, betas, c2w, focal, h, w, pp, m_rgb, m_bs, m_attn, scale_factor=1.0, eps2d=0.3, deconf=False):
    from gsplat.rendering import rasterization

    n = means.shape[0]
    cx, cy = (0.5 * w, 0.5 * h) if pp is None else (float(pp[0]), float(pp[1]))
    k = torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], device=means.device, dtype=means.dtype)
    if not torch.is_tensor(betas):
        betas = torch.full((n,), float(betas), device=means.device, dtype=means.dtype)
    out, alpha, meta = rasterization(
        means=means, l_triagnles=quats, scales=scales * scale_factor, opacities=opacity.reshape(n), betas=betas.reshape(n),
        colors=colors.clamp_min(0.0), viewmats=torch.linalg.inv(c2w)[None], Ks=k[None], width=w, height=h, near_plane=0.2,
        eps2d=eps2d, render_mode="RGB", medium_rgb=m_rgb[None], medium_bs=m_bs[None], medium_attn=m_attn[None], **(SGA_STAGE2 if deconf else {}))
    img = (out[0][..., :3] + meta["render_medium"][0]).permute(2, 0, 1)
    clear = meta["render_clear"][0].permute(2, 0, 1) if meta.get("render_clear") is not None else None
    return img, alpha[0].permute(2, 0, 1), clear


def reference_compositor(means2d, radius2, depth, opacity, betas, colors, m_rgb, m_bs, m_attn, h, w, attn_alpha=True, cap=1.0):
    dev = means2d.device
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=means2d.dtype) + 0.5, torch.arange(w, device=dev, dtype=means2d.dtype) + 0.5,
                            indexing="ij")
    T = torch.ones(h, w, device=dev, dtype=means2d.dtype)
    out = torch.zeros(h, w, 3, device=dev, dtype=means2d.dtype)
    clear = torch.zeros_like(out)
    med = torch.zeros_like(out)
    prev = torch.zeros(h, w, device=dev, dtype=means2d.dtype)
    done = torch.zeros(h, w, dtype=torch.bool, device=dev)
    for i in torch.argsort(depth).tolist():
        sig = ((xs - means2d[i, 0]) ** 2 + (ys - means2d[i, 1]) ** 2) / radius2[i]
        hit = (sig < 1.0) & ~done
        a_raw = torch.clamp(opacity[i] * (1.0 - sig.clamp(0.0, 1.0)) ** betas[i], max=0.999)
        g = 0.5 + 0.5 * torch.exp(-m_attn * depth[i]).mean(-1) if attn_alpha else torch.ones_like(T)
        a = torch.where(hit, a_raw * g, torch.zeros_like(T))
        nxt = T * (1.0 - a)
        stop = hit & (nxt <= 1e-4)
        done = done | stop
        use = hit & ~stop
        vis = torch.where(use, a * T, torch.zeros_like(T))[..., None]
        out = out + torch.exp(-m_attn * depth[i]) * colors[i] * vis
        clear = clear + colors[i] * vis
        seg = T[..., None] * (torch.exp(-m_bs * prev[..., None]) - torch.exp(-m_bs * depth[i])) * m_rgb
        med = torch.where(use[..., None], (med + seg).clamp(max=cap), med)
        prev = torch.where(use, torch.full_like(prev, float(depth[i])), prev)
        T = torch.where(use, nxt, T)
    med = (med + T[..., None] * torch.exp(-m_bs * prev[..., None]) * m_rgb).clamp(max=cap)
    return out, 1.0 - T, med, clear
