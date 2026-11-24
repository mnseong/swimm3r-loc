"""Anchor head: fewer, larger Gaussians for the feed forward map, on the features of the Splatt3R head.

After AnchorSplat (arXiv 2604.07053, anchors from a 3D prior) and the voxel feature lifting of VolSplat
(arXiv 2509.19297). The pixels that pass the confidence filter are lifted to 3D at our own depth and grouped in
voxels whose edge is F pixel footprints, the footprint rounded to a power of two of the median footprint as in
voxel_fuse, so near and far surfaces get the same number of pixels per anchor. One anchor per voxel with
- position: the weighted centroid of its points, weights log(1 + confidence) on the Stage 1 confidence scale,
- surface frame: the normal is the weighted mean of the per pixel normals of the depth maps (pixel_normals),
  turned towards the cameras, so a small depth disagreement between views does not tip it (the viewing ray when
  no pixel has a normal); the first tangent is the mean x axis of the cameras that saw the anchor projected
  into the surface, so the frame turns with the cameras and a predicted tilt means the same in every map,
- input: the weighted mean of the 128 channel features of the Splatt3R DPT (the layer before its last
  convolution), mean and spread of the restored colour J, the extents of the points along the frame axes in
  voxel units, the cosine between normal and viewing ray and the pixels per view per F^2. None of these depends
  on the frame of the coordinates or on the number of views, so a window in the anchor camera (training) and a
  whole Stage 1 map (inference) look alike.
Two residual blocks mix every anchor with the mean of its occupied 26 neighbours. An MLP decodes one Gaussian
per anchor in its surface frame: offset along the normal bounded to half an edge (the centre stays at the
centroid in the plane), scales = the extents times exp(prediction) (no bound and no loss on the size), rotation =
surface frame times a predicted residual, colour = mean J plus an SH residual as Splatt3R, opacity sigmoid.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.splatt3r_head import Splatt3RHead
from training.gaussian_branch import quat_mul, rotmat_to_quat

FEAT_DIM = 128
N_OUT = 14
LEVELS = 4
CONF_SHIFT = 0.999
_OFF = torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def canonical_conf(conf):
    """Pair pass confidence of the pointmap head (1 + exp) on the scale of the Stage 1 confidence, which
    canonical_view of mast3r/cloud_opt/sparse_ga.py forms from conf - 0.999 of the pairs."""
    return (conf - CONF_SHIFT).clamp_min(0.0)


def pixel_normals(pts, z, tol=0.1):
    """Unit normals (K, H, W, 3) of the points (K, H, W, 3) of K depth maps from central differences, the border
    taken from the nearest inner pixel, zero at depth steps above tol of the depth across two pixels and where the
    depth is not valid."""
    du = pts[:, 1:-1, 2:] - pts[:, 1:-1, :-2]
    dv = pts[:, 2:, 1:-1] - pts[:, :-2, 1:-1]
    c = torch.cross(du, dv, dim=-1)
    zc = z[:, 1:-1, 1:-1]
    smooth = ((z[:, 1:-1, 2:] - z[:, 1:-1, :-2]).abs() < tol * zc) & ((z[:, 2:, 1:-1] - z[:, :-2, 1:-1]).abs() < tol * zc)
    ok = smooth & torch.isfinite(c).all(-1) & (c.norm(dim=-1) > 0) & (zc > 1e-4)
    inner = F.normalize(torch.nan_to_num(c), dim=-1)
    inner = torch.where(ok[..., None], inner, torch.zeros_like(inner))
    return F.pad(inner.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1).contiguous()


def _pack(level, c):
    c = (c + (1 << 18)).clamp(0, (1 << 19) - 1)
    return ((level + LEVELS) << 57) | (c[..., 0] << 38) | (c[..., 1] << 19) | c[..., 2]


@torch.no_grad()
def _frames(nsum, view, xsum):
    """Surface frame (M, 3, 3), columns tangent 1, tangent 2, normal, from the summed pixel normals, the mean
    viewing ray and the summed camera x axes."""
    m = nsum.shape[0]
    eye = torch.eye(3, device=nsum.device, dtype=nsum.dtype)
    has = nsum.norm(dim=-1) > 0.3
    n = torch.where(has[:, None], F.normalize(nsum, dim=-1), -view)
    n = torch.where(((n * view).sum(-1) > 0)[:, None], -n, n)
    t1 = xsum - (xsum * n).sum(-1, keepdim=True) * n
    weak = t1.norm(dim=-1) < 0.3 * xsum.norm(dim=-1).clamp_min(1e-9)
    ref = torch.where((n[:, 0].abs() < 0.9)[:, None], eye[0].expand(m, 3), eye[1].expand(m, 3))
    t1f = ref - (ref * n).sum(-1, keepdim=True) * n
    t1 = F.normalize(torch.where(weak[:, None], t1f, t1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], dim=-1)


def build_anchors(xyz, fp, conf, cam, camx, nrm, vid, feat, J, raw, anchor_px, quadrants=False, cells=2, times=None):
    """Anchors of the lifted pixels. xyz (N, 3) points, fp (N) footprints (depth / focal), conf (N) confidence on
    the Stage 1 scale, cam and camx (N, 3) centre and x axis of the camera of the pixel, nrm (N, 3) pixel normal
    (zero where none), vid (N) view index, feat (N, C), J and raw (N, 3) restored and underwater colours. Features
    and colours keep their gradient, the geometry has none. With quadrants also the cells x cells cells of every
    anchor in its tangent plane: the coordinates (u, v) along tangent 1 and 2 about the centroid, in edges, binned at
    -0.5 + k / cells (cells 2 splits at the centroid), cell index iu + cells iv: q_count (M, K) pixels, q_share (M, K)
    share of the anchor weight, q_uvn and q_sd (M, K, 3) weighted mean and spread of the frame coordinates in edges,
    q_feat (M, K, C), q_J and q_raw (M, K, 3), K = cells^2. With times (N,), the capture time of every pixel in
    seconds, also t (M,) and with quadrants q_t (M, K), the weighted mean times of the anchors and of their cells."""
    with torch.no_grad():
        fp_ref = fp.median().clamp_min(1e-9)
        level = torch.round(torch.log2((fp / fp_ref).clamp_min(1e-6))).clamp(-LEVELS, LEVELS).long()
        cell = float(anchor_px) * fp_ref * torch.pow(2.0, level.float())
        c = torch.floor(xyz / cell[:, None]).long()
        key = _pack(level, c)
        uniq, inv = torch.unique(key, return_inverse=True)
        m = uniq.numel()
        w = torch.log1p(conf.clamp(0.0, 1e4)).clamp_min(1e-6)
        wsum = torch.zeros(m, device=xyz.device, dtype=w.dtype).index_add_(0, inv, w)
        wn = w / wsum[inv]
        count = torch.zeros(m, device=xyz.device).index_add_(0, inv, torch.ones_like(w))
        nv = int(vid.max()) + 1
        pairs = torch.unique(inv * nv + vid.long())
        views = torch.zeros(m, device=xyz.device).index_add_(0, pairs // nv, torch.ones(pairs.numel(), device=xyz.device))
        edge = torch.zeros(m, device=xyz.device, dtype=cell.dtype).scatter_(0, inv, cell)
        lev = torch.zeros(m, device=xyz.device, dtype=torch.long).scatter_(0, inv, level)
        coord = torch.zeros(m, 3, device=xyz.device, dtype=torch.long).scatter_(0, inv[:, None].expand(-1, 3), c)

        def wmean(a):
            return torch.zeros((m,) + a.shape[1:], device=a.device, dtype=a.dtype).index_add_(0, inv, a * wn.to(a.dtype).view((-1,) + (1,) * (a.dim() - 1)))

        centroid = wmean(xyz)
        d = (xyz - centroid[inv]) / edge[inv][:, None]
        cov = wmean(d[:, :, None] * d[:, None, :])
        view = F.normalize(wmean(F.normalize(xyz - cam, dim=-1)), dim=-1)
        ray = F.normalize(xyz - cam, dim=-1)
        nrm = torch.where(((nrm * ray).sum(-1) > 0)[:, None], -nrm, nrm)
        frame = _frames(wmean(nrm), view, wmean(camx))
        var = torch.einsum("mia,mij,mja->ma", frame, cov, frame).clamp_min(0.0)
        density = (count / (views.clamp_min(1.0) * float(anchor_px) ** 2)).clamp(0.0, 4.0)
        nbr = []
        for o in _OFF.to(xyz.device):
            k = _pack(lev, coord + o)
            idx = torch.searchsorted(uniq, k).clamp(max=m - 1)
            nbr.append(torch.where(uniq[idx] == k, idx, torch.full_like(idx, -1)))
        nbr = torch.stack(nbr, dim=1)

    def mean(a):
        return torch.zeros((m,) + a.shape[1:], device=a.device, dtype=torch.float32).index_add_(
            0, inv, a.float() * wn.view((-1,) + (1,) * (a.dim() - 1)))

    jm = mean(J)
    jsd = (mean(J * J) - jm * jm).clamp_min(0.0).add(1e-8).sqrt()
    out = dict(centroid=centroid, edge=edge, frame=frame, var=var, view=view, count=count, views=views, density=density,
               nbr=nbr, feat=mean(feat), J=jm, J_sd=jsd, raw=mean(raw), n_pix=int(xyz.shape[0]))
    if times is not None:
        with torch.no_grad():
            out["t"] = mean(times.float()[:, None])[:, 0]
    if quadrants:
        with torch.no_grad():
            K = cells * cells
            loc = torch.einsum("nij,ni->nj", frame[inv], d)
            iu = torch.floor((loc[:, 0] + 0.5) * cells).clamp(0, cells - 1).long()
            iv = torch.floor((loc[:, 1] + 0.5) * cells).clamp(0, cells - 1).long()
            key = inv * K + iu + cells * iv
            wq_sum = torch.zeros(K * m, device=xyz.device, dtype=w.dtype).index_add_(0, key, w)
            wq = w / wq_sum[key]
            qn = torch.zeros(K * m, device=xyz.device).index_add_(0, key, torch.ones_like(w))
            quvn = torch.zeros(K * m, 3, device=xyz.device, dtype=loc.dtype).index_add_(0, key, loc * wq[:, None].to(loc.dtype))
            q2 = torch.zeros(K * m, 3, device=xyz.device, dtype=loc.dtype).index_add_(0, key, loc * loc * wq[:, None].to(loc.dtype))
            qsd = (q2 - quvn * quvn).clamp_min(0.0).sqrt()

        def qmean(a):
            return torch.zeros((K * m,) + a.shape[1:], device=a.device, dtype=torch.float32).index_add_(
                0, key, a.float() * wq.view((-1,) + (1,) * (a.dim() - 1))).view((m, K) + a.shape[1:])

        out.update(q_count=qn.view(m, K), q_share=(wq_sum.view(m, K) / wsum[:, None]).float(), q_uvn=quvn.view(m, K, 3).float(),
                   q_sd=qsd.view(m, K, 3).float(), q_feat=qmean(feat), q_J=qmean(J), q_raw=qmean(raw), cells=cells)
        if times is not None:
            with torch.no_grad():
                out["q_t"] = qmean(times.float()[:, None])[..., 0]
    return out


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, f, nbr):
        h = self.norm(f)
        hz = torch.cat([h, h.new_zeros(1, h.shape[1])])
        ok = (nbr >= 0).to(h.dtype)
        s = h.new_zeros(h.shape)
        for j in range(nbr.shape[1]):
            s = s + hz[nbr[:, j]]
        s = s / ok.sum(1, keepdim=True).clamp_min(1.0)
        return f + self.mlp(torch.cat([h, s], dim=-1))


class AnchorHead(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, width=128, blocks=2, opacity_bias=2.0):
        super().__init__()
        in_dim = feat_dim + 3 + 3 + 3 + 1 + 1
        self.cfg = dict(feat_dim=feat_dim, width=width, blocks=blocks, opacity_bias=opacity_bias)
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([_Block(width) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, N_OUT))
        last = self.out[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.zero_()
            last.bias[6] = 1.0
            last.bias[13] = opacity_bias

    def context(self, a):
        """Context vector (M, width) of every anchor after the neighbour blocks, and the spread sd (M, 3) of its points."""
        sd = a["var"].sqrt()
        cosv = (a["frame"][..., 2] * a["view"]).sum(-1, keepdim=True).abs()
        x = torch.cat([a["feat"].float(), a["J"], a["J_sd"], sd, cosv, a["density"][:, None]], dim=-1)
        f = self.inp(x)
        for b in self.blocks:
            f = b(f, a["nbr"])
        return f, sd

    def forward(self, a):
        """Gaussians of the anchors of build_anchors: means (M, 3), scales (M, 3), quat (M, 4) w x y z, opacity (M),
        J (M, 3) restored colour, raw (M, 3) mean underwater colour of the anchor, offset (M, 3) and scale_edges (M)
        the largest axis, both in edges."""
        f, sd = self.context(a)
        return self.decode(a, f, sd)

    def decode(self, a, f, sd):
        o = self.out(f)
        off = 0.5 * torch.tanh(o[:, 0:3]) * o.new_tensor([0.0, 0.0, 1.0])
        e = a["edge"][:, None]
        base = torch.stack([sd[:, 0].clamp(0.15, 0.6), sd[:, 1].clamp(0.15, 0.6), sd[:, 2].clamp(0.02, 0.3)], dim=-1)
        rel = base * torch.exp(o[:, 3:6])
        means = a["centroid"] + e * torch.einsum("mij,mj->mi", a["frame"], off)
        quat = quat_mul(rotmat_to_quat(a["frame"]), F.normalize(o[:, 6:10], dim=-1))
        colour = Splatt3RHead.colour(o[:, 10:13], a["J"])
        return dict(means=means, scales=e * rel, quat=quat, opacity=torch.sigmoid(o[:, 13]), J=colour, raw=a["raw"], offset=off,
                    scale_edges=rel.detach().max(dim=-1).values)


BETA_BASE, BETA_RANGE = 4.0, 2.0
GATE_P0 = 0.04742587317756678
TW_BASE, TW_MIN, TW_MAX = 1.5, -1.5, 3.0
TB_RANGE = 2.0
VEL_UNIT_EDGE, VEL_RANGE = 10.0 / 8.0, 20.0
N_X = 7
PRUNE_OPACITY = 1.0 / 255.0


def activate_x(raw, gate_init=0.0):
    """Beta shape, dynamic gate, relative temporal width, temporal shape and velocity of raw (..., 7), the
    activations of the 21 channel head of model/gaussian_head.py (the same constants), with the velocity in edges per
    second in the anchor frame (10 footprints per second per unit there, an edge of 8 footprints here). A zero raw
    gives beta 4 and a static slot (gate 0). With gate_init > 0 the gate is sigmoid(raw + logit(gate_init)) instead:
    gate_init for a zero raw, always inside (0, 1), so its gradient does not stop below 0 as that of the 21 channel
    form does behind the clamp of slot_at_time."""
    if gate_init > 0:
        gate = torch.sigmoid(raw[..., 1] + math.log(gate_init / (1.0 - gate_init)))
    else:
        u = torch.expm1(raw[..., 1].clamp(-20.0, 20.0))
        gate = GATE_P0 * u / (1.0 + GATE_P0 * u)
    return dict(beta=BETA_BASE * torch.exp(raw[..., 0].clamp(-BETA_RANGE, BETA_RANGE)), gate=gate,
                tw_rel=TW_BASE * torch.exp(raw[..., 2].clamp(TW_MIN, TW_MAX)), tb=raw[..., 3].clamp(-TB_RANGE, TB_RANGE),
                vel_e=VEL_UNIT_EDGE * raw[..., 4:7].clamp(-VEL_RANGE, VEL_RANGE))


def _floor(x, lo):
    """Smooth lower bound sqrt(lo^2 + x^2) >= lo."""
    return torch.sqrt(x * x + lo * lo)


class ScaffoldHead(nn.Module):
    """Anchor head with children (Scaffold-GS style), so detail finer than a voxel can be kept where it pays.

    Every anchor has up to five slots. The parent is the Gaussian of AnchorHead (the module self.parent, so the
    weights of a trained anchor head load into it unchanged). The children sit in the four quadrants of the anchor's
    tangent plane, or with cells 3 in its 3 x 3 cells (build_anchors(quadrants=True, cells=...)); cells 3 gives ten
    slots per anchor, the n_offsets default of Scaffold-GS. A child reads the anchor context of the parent's
    neighbour blocks and the weighted mean feature, restored colour, spread, centroid and share of its own cell, so it
    sees where in the voxel the texture is. A child starts at its cell centroid, moves at most half a cell in the
    plane and half an edge along the normal, takes its scales from the spread of its cell (clamped to 0.16 / cells to
    0.6 / cells edges) times exp(prediction) and its colour from the cell's restored colour plus a residual. Empty
    cells have no child.
    Every slot has a use probability p. mode 'soft' renders opacity p, 'hard' the step (p > 0.5) with the gradient
    of p (straight through), 'infer' keeps the slots with p > 0.5 only, 'infer_soft' renders as 'soft' and leaves out
    the slots whose opacity p is not above PRUNE_OPACITY (the map a soft trained head was trained on). use_mean is the
    mean p over the slots, the sparsity term of the loss. With always_on there is no use probability, as in
    Scaffold-GS: every slot whose opacity is above PRUNE_OPACITY is drawn, by the same rule in training and in the
    map, so mode does not matter, and the children start faint (opacity sigmoid(child_use_bias)) where p used to hold
    them back. The in plane axes of parent and children have a smooth floor (parent_floor and
    child_floor edges) so that neighbours touch (the gaps seen in the renders of the anchor head).
    With beta or four_d every slot also has the seven extra outputs of activate_x (zero initialised, so the head
    starts as the Gaussian static scaffold with beta 4): beta (P,) for the Beta kernel, and gate, tw_rel, tb, vel
    (P, 3, world units per second, the velocity in the anchor frame times the edge) and t_src (P,), the mean capture
    time of the pixels of the slot (anchor or cell), for the time split of slot_at_time. gate_init > 0 starts every
    gate at gate_init with the sigmoid form of activate_x. beta_fixed > 0 (needs beta) gives every slot that constant
    beta instead of the output of the head, 4 being the shape fitted to the Gaussian (training/primitives4d.py
    K_BETA4): the same rasteriser and head without a learned kernel shape."""

    scaffold = True

    def __init__(self, feat_dim=FEAT_DIM, width=128, blocks=2, opacity_bias=2.0, child_use_bias=-2.0, parent_use_bias=3.0,
                 parent_floor=0.35, child_floor=0.18, cells=2, beta=False, four_d=False, scaffold=True, always_on=False,
                 gate_init=0.0, beta_fixed=0.0):
        super().__init__()
        if not 0.0 <= float(gate_init) < 1.0:
            raise ValueError(f"gate_init {gate_init} outside [0, 1)")
        if float(beta_fixed) < 0.0:
            raise ValueError(f"beta_fixed {beta_fixed} below 0")
        if float(beta_fixed) > 0.0 and not beta:
            raise ValueError("beta_fixed needs beta (the Beta rasteriser)")
        self.parent = AnchorHead(feat_dim=feat_dim, width=width, blocks=blocks, opacity_bias=opacity_bias)
        self.cfg = dict(feat_dim=feat_dim, width=width, blocks=blocks, opacity_bias=opacity_bias, child_use_bias=child_use_bias,
                        parent_use_bias=parent_use_bias, parent_floor=parent_floor, child_floor=child_floor, cells=cells,
                        beta=bool(beta), four_d=bool(four_d), scaffold=True, always_on=bool(always_on), gate_init=float(gate_init),
                        beta_fixed=float(beta_fixed))
        self.cells = int(cells)
        self.extra = N_X if (beta or four_d) else 0
        cin = width + feat_dim + 3 + 3 + 3 + 1 + self.cells ** 2
        self.child_inp = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, width), nn.GELU(), nn.Linear(width, width))
        self.child_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, N_OUT + 1 + self.extra))
        self.parent_use = nn.Linear(width, 1)
        self.parent_x = nn.Linear(width, N_X) if self.extra else None
        with torch.no_grad():
            last = self.child_out[-1]
            last.weight.mul_(0.01)
            last.bias.zero_()
            last.bias[6] = 1.0
            last.bias[13] = child_use_bias if always_on else opacity_bias
            last.bias[14] = child_use_bias
            self.parent_use.weight.zero_()
            self.parent_use.bias.fill_(parent_use_bias)
            if self.parent_x is not None:
                self.parent_x.weight.zero_()
                self.parent_x.bias.zero_()
        self.mode = "soft"

    def forward(self, a):
        P = self.parent
        f, sd = P.context(a)
        po = P.decode(a, f, sd)
        e = a["edge"][:, None]
        rel = po["scales"] / e
        rel = torch.cat([_floor(rel[:, :2], self.cfg["parent_floor"]), rel[:, 2:]], dim=-1)
        means, scales, quat, opac, Js, raws = [po["means"]], [e * rel], [po["quat"]], [po["opacity"]], [po["J"]], [po["raw"]]
        use = [torch.sigmoid(self.parent_use(f)[:, 0])]
        slot = [torch.zeros(f.shape[0], dtype=torch.long, device=f.device)]
        n_child = 0
        X = []
        if self.extra:
            px = activate_x(self.parent_x(f), self.cfg["gate_init"])
            vel = e * torch.einsum("mij,mj->mi", a["frame"], px["vel_e"])
            t0 = a["t"] if "t" in a else torch.zeros_like(a["edge"])
            X.append(dict(beta=px["beta"], gate=px["gate"], tw_rel=px["tw_rel"], tb=px["tb"], vel=vel, t_src=t0.to(vel.dtype)))
        if "q_feat" in a:
            m, K, s = f.shape[0], self.cells ** 2, self.cells
            if a["q_count"].shape[1] != K:
                raise ValueError(f"anchors pooled on {a['q_count'].shape[1]} cells, the head has {s} x {s}")
            onehot = torch.eye(K, device=f.device, dtype=f.dtype)[None].expand(m, K, K)
            xc = torch.cat([f[:, None].expand(m, K, f.shape[1]), a["q_feat"].to(f.dtype), a["q_J"], a["q_sd"], a["q_uvn"],
                            a["q_share"][..., None], onehot], dim=-1)
            oc = self.child_out(self.child_inp(xc))
            exists = (a["q_count"] > 0).reshape(-1)
            delta = torch.cat([(0.5 / s) * torch.tanh(oc[..., 0:2]), 0.5 * torch.tanh(oc[..., 2:3])], dim=-1)
            loc = a["q_uvn"] + delta
            ec = e[:, :, None]
            mc = a["centroid"][:, None] + ec * torch.einsum("mij,mqj->mqi", a["frame"], loc)
            lo, hi = 0.16 / s, 0.6 / s
            base = torch.stack([a["q_sd"][..., 0].clamp(lo, hi), a["q_sd"][..., 1].clamp(lo, hi),
                                a["q_sd"][..., 2].clamp(0.02, 0.3)], dim=-1)
            relc = base * torch.exp(oc[..., 3:6])
            relc = torch.cat([_floor(relc[..., :2], self.cfg["child_floor"]), relc[..., 2:]], dim=-1)
            qc = quat_mul(rotmat_to_quat(a["frame"])[:, None].expand(m, K, 4).reshape(-1, 4),
                          F.normalize(oc[..., 6:10], dim=-1).reshape(-1, 4))
            jc = Splatt3RHead.colour(oc[..., 10:13], a["q_J"])
            means.append(mc.reshape(-1, 3)[exists])
            scales.append((ec * relc).reshape(-1, 3)[exists])
            quat.append(qc[exists])
            opac.append(torch.sigmoid(oc[..., 13]).reshape(-1)[exists])
            Js.append(jc.reshape(-1, 3)[exists])
            raws.append(a["q_raw"].reshape(-1, 3)[exists])
            use.append(torch.sigmoid(oc[..., 14]).reshape(-1)[exists])
            slot.append((torch.arange(K, device=f.device) + 1)[None].expand(m, K).reshape(-1)[exists])
            n_child = int(exists.sum())
            if self.extra:
                cx = activate_x(oc[..., N_OUT + 1:N_OUT + 1 + N_X], self.cfg["gate_init"])
                velc = ec * torch.einsum("mij,mqj->mqi", a["frame"], cx["vel_e"])
                tq = a["q_t"] if "q_t" in a else torch.zeros_like(a["q_share"])
                X.append(dict(beta=cx["beta"].reshape(-1)[exists], gate=cx["gate"].reshape(-1)[exists],
                              tw_rel=cx["tw_rel"].reshape(-1)[exists], tb=cx["tb"].reshape(-1)[exists],
                              vel=velc.reshape(-1, 3)[exists], t_src=tq.reshape(-1)[exists].to(velc.dtype)))
        means, scales, quat, opac = torch.cat(means), torch.cat(scales), torch.cat(quat), torch.cat(opac)
        Js, raws, use, slot = torch.cat(Js), torch.cat(raws), torch.cat(use), torch.cat(slot)
        edge_all = torch.cat([a["edge"], a["edge"][:, None].expand(-1, self.cells ** 2).reshape(-1)[exists]]) if n_child else a["edge"]
        on = self.cfg.get("always_on", False)
        if on:
            opacity = opac
            sel = opac.detach() > PRUNE_OPACITY
        elif self.mode == "infer":
            opacity = opac
            sel = use.detach() > 0.5
        elif self.mode == "infer_soft":
            opacity = opac * use
            sel = opacity.detach() > PRUNE_OPACITY
        elif self.mode == "hard":
            opacity = opac * ((use > 0.5).to(use.dtype) + use - use.detach())
            sel = None
        else:
            opacity = opac * use
            sel = None
        kept = sel if sel is not None else use.detach() > 0.5
        stats = dict(n_anchor=int(f.shape[0]), n_child=n_child, parent_kept=int(kept[slot == 0].sum()),
                     child_kept=int(kept[slot > 0].sum()))
        if not on:
            stats["use_mean"] = use.mean()
        xs = {k: torch.cat([x[k] for x in X]) for k in X[0]} if X else {}
        if sel is not None:
            means, scales, quat, opacity, Js, raws, edge_all, use, slot = (
                t[sel] for t in (means, scales, quat, opacity, Js, raws, edge_all, use, slot))
            xs = {k: v[sel] for k, v in xs.items()}
        bf = float(self.cfg.get("beta_fixed", 0.0) or 0.0)
        if bf > 0.0 and "beta" in xs:
            xs["beta"] = torch.full_like(xs["beta"], bf)
        if not self.cfg["beta"]:
            xs.pop("beta", None)
        if not self.cfg["four_d"]:
            xs = {k: v for k, v in xs.items() if k == "beta"}
        extra = dict(slot=slot) if on else dict(use=use, slot=slot)
        return dict(means=means, scales=scales, quat=quat, opacity=opacity, J=Js, raw=raws, offset=po["offset"],
                    scale_edges=(scales.detach() / edge_all[:, None]).max(dim=-1).values, **extra, **stats, **xs)


def slot_at_time(g, t, min_opacity=1.0 / 255.0):
    """The slots of a 4D scaffold map at time t (seconds, the time base of t_src): a static copy with opacity
    o (1 - gate) and, where o gate > min_opacity, a dynamic copy with opacity o gate window(t - t_src) moved by
    vel (t - t_src), window the temporal opacity of Universal Beta Splatting (training/primitives4d.py). g holds
    means, opacity, scales, quat, J and beta, gate, tw (seconds), tb, vel, t_src per slot; without gate or with t None
    g is returned as it is. The copies keep the gradient of every input."""
    if t is None or "gate" not in g:
        return g
    from training.primitives4d import temporal_window
    gate = g["gate"].clamp(0.0, 1.0)
    o = g["opacity"]
    dyn = (o * gate).detach() > min_opacity
    dt = (torch.as_tensor(t, dtype=g["t_src"].dtype, device=g["t_src"].device) - g["t_src"])[dyn]
    out = dict(g)
    out["opacity"] = torch.cat([o * (1.0 - gate), (o * gate)[dyn] * temporal_window(dt, g["tw"][dyn], g["tb"][dyn])])
    out["means"] = torch.cat([g["means"], g["means"][dyn] + g["vel"][dyn] * dt[:, None]])
    for k in ("scales", "quat", "J", "beta"):
        if k in g:
            out[k] = torch.cat([g[k], g[k][dyn]])
    out["n_dyn"] = int(dyn.sum())
    return out


def describe(m):
    return f"anchor head {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters, {m.cfg}"
