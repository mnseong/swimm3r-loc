import time

import numpy as np
import torch
import torch.nn.functional as F

import loc_geom as G


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def encode_frames(net, imgs, chunk=4):
    n, _, h, w = imgs.shape
    shape = torch.tensor([[h, w]], device=imgs.device)
    out = []
    for s in range(0, n, chunk):
        f, _, _ = net._encode_image(imgs[s:s + chunk] * 2.0 - 1.0, shape.repeat(min(chunk, n - s), 1))
        out.append(f.float())
    return torch.cat(out)


def desc_only(net, num, dec, hw):
    head = net.downstream_head1 if num == 1 else net.downstream_head2
    x = torch.cat([dec[0], dec[-1]], dim=-1)
    ps = int(head.patch_size)
    f = head.head_local_features(x).transpose(-1, -2).reshape(x.shape[0], -1, hw[0] // ps, hw[1] // ps)
    f = F.pixel_shuffle(f, ps).permute(0, 2, 3, 1)[..., :int(head.local_feat_dim)]
    return f / f.norm(dim=-1, keepdim=True)


class Localizer:
    def __init__(self, net, tokens, depth, conf, intr, c2w, focal, pp, min_conf, dev, subsample=8, reproj=None, name="ours", points=2000):
        n, h, w = depth.shape
        self.net, self.tokens, self.dev, self.name = net, tokens, dev, name
        self.n, self.h, self.w = n, h, w
        self.shape = torch.tensor([[h, w]], device=dev)
        v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        f = intr[:, 0, 0].float()[:, None, None]
        cam = torch.stack([(u - intr[:, 0, 2].float()[:, None, None]) / f * depth, (v - intr[:, 1, 2].float()[:, None, None]) / f * depth, depth], -1)
        self.xyz = cam @ c2w[:, :3, :3].float().transpose(1, 2)[:, None] + c2w[:, None, None, :3, 3].float()
        self.valid = torch.isfinite(self.xyz).all(-1) & (depth > 1e-4) & (conf > min_conf)
        self.c2w = c2w.float()
        self.poses = c2w.detach().cpu().numpy().astype(np.float64)
        self.centres, self.forward = self.c2w[:, :3, 3], self.c2w[:, :3, 2]
        kd, pts = [], []
        gen = torch.Generator().manual_seed(0)
        for i in range(n):
            m = self.valid[i]
            kd.append(depth[i][m].median() if bool(m.any()) else depth[i].flatten().median())
            p = self.xyz[i][m]
            if p.shape[0] > points:
                p = p[torch.randperm(p.shape[0], generator=gen)[:points].to(dev)]
            pts.append(p.double().cpu().numpy())
        self.kd = torch.stack(kd).float().clamp_min(1e-9)
        self.depths = self.kd.cpu().numpy().astype(np.float64)
        self.points = pts
        self.focal, self.pp = float(focal), (float(pp[0]), float(pp[1]))
        self.K = G.intrinsics(focal, pp)
        self.mean_token = tokens.float().mean((0, 1))
        self.subsample = int(subsample)
        self.reproj = float(reproj) if reproj else 0.008 * float(np.hypot(h, w))
        self.pos = None
        self.lean_ok, self.lean_diff, self.half_ok = None, None, None

    def amp(self, half):
        return torch.autocast("cuda", dtype=torch.float16, enabled=bool(half) and self.dev.type == "cuda")

    def descriptors(self, d1, d2, sh, lean):
        d1, d2 = [t.float() for t in d1], [t.float() for t in d2]
        hw = (self.h, self.w)
        if lean and self.lean_ok is None:
            try:
                if self.h > self.w:
                    raise ValueError("portrait frames go through the full head")
                with torch.autocast("cuda", enabled=False):
                    full = self.net._downstream_head(1, d1, sh)["desc"], self.net._downstream_head(2, d2, sh)["desc"]
                    mine = desc_only(self.net, 1, d1, hw), desc_only(self.net, 2, d2, hw)
                self.lean_diff = float(max((a - b).abs().max() for a, b in zip(full, mine)))
                self.lean_ok = bool(self.lean_diff < 1e-3)
                print("[loc] %s: descriptors without the pointmap branch differ from the full head by at most %.1e, %s" % (
                    self.name, self.lean_diff, "used from here on" if self.lean_ok else "not used"), flush=True)
                return full
            except Exception as e:
                self.lean_ok = False
                print("[loc] %s: descriptors without the pointmap branch are not available (%s: %s), the full head is used" % (
                    self.name, type(e).__name__, e), flush=True)
        with torch.autocast("cuda", enabled=False):
            if lean and self.lean_ok:
                return desc_only(self.net, 1, d1, hw), desc_only(self.net, 2, d2, hw)
            return self.net._downstream_head(1, d1, sh)["desc"], self.net._downstream_head(2, d2, sh)["desc"]

    def sub(self, keep):
        import copy
        keep = [int(i) for i in keep]
        ii = torch.as_tensor(keep, device=self.dev, dtype=torch.long)
        o = copy.copy(self)
        o.n = len(keep)
        o.tokens, o.xyz, o.valid, o.c2w = self.tokens[ii], self.xyz[ii], self.valid[ii], self.c2w[ii]
        o.poses, o.depths, o.kd = self.poses[keep], self.depths[keep], self.kd[ii]
        o.centres, o.forward = o.c2w[:, :3, 3], o.c2w[:, :3, 2]
        o.points = [self.points[i] for i in keep]
        o.pos = None
        return o

    def megabytes(self):
        return {"tokens": self.tokens.numel() * self.tokens.element_size() / 1e6, "points": self.xyz.numel() * 4 / 1e6}

    @torch.no_grad()
    def encode(self, img, half=False):
        with self.amp(half):
            f, pos, _ = self.net._encode_image(img[None] * 2.0 - 1.0, self.shape)
        self.pos = pos
        return f

    def near(self, pose, k=1):
        p = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=self.dev)
        dt = (self.centres - p[:3, 3]).norm(dim=-1) / self.kd
        ang = torch.acos((self.forward @ p[:3, 2]).clamp(-1.0, 1.0))
        return torch.argsort(dt + ang)[:k].tolist()

    @torch.no_grad()
    def retrieve(self, fq, k=1, chunk=16):
        q = F.normalize(fq[0].float() - self.mean_token, dim=-1)
        out = []
        for s in range(0, self.n, chunk):
            t = F.normalize(self.tokens[s:s + chunk].float() - self.mean_token, dim=-1)
            sim = torch.einsum("nc,kmc->knm", q, t)
            out.append(0.5 * (sim.max(2).values.mean(1) + sim.max(1).values.mean(1)))
        score = torch.cat(out)
        return torch.argsort(score, descending=True)[:k].tolist(), score

    @torch.no_grad()
    def match(self, fq, idx, lean=False, half=False):
        from mast3r.fast_nn import fast_reciprocal_NNs
        k = len(idx)
        ii = torch.as_tensor(idx, device=self.dev, dtype=torch.long)
        pos = self.pos.expand(k, -1, -1)
        t0 = time.time()
        with self.amp(half):
            d1, d2 = self.net._decoder(fq.expand(k, -1, -1), pos, self.tokens[ii].to(fq.dtype), pos)
        sh = self.shape.expand(k, 2)
        desc1, desc2 = self.descriptors(d1, d2, sh, lean)
        sync(self.dev)
        t1 = time.time()
        p2d, p3d, found = [], [], 0
        for j, i in enumerate(idx):
            xm, xq = fast_reciprocal_NNs(desc2[j], desc1[j], subsample_or_initxy1=self.subsample, ret_xy=True,
                                         device=self.dev, dist="dot", block_size=2 ** 13)
            found += len(xm)
            if len(xm) == 0:
                continue
            xm = torch.as_tensor(np.ascontiguousarray(xm), device=self.dev, dtype=torch.long)
            ok = self.valid[i][xm[:, 1], xm[:, 0]]
            p3d.append(self.xyz[i][xm[:, 1], xm[:, 0]][ok].double().cpu().numpy())
            p2d.append(np.ascontiguousarray(xq, dtype=np.float64)[ok.cpu().numpy()])
        sync(self.dev)
        t2 = time.time()
        if not p2d:
            return np.zeros((0, 2)), np.zeros((0, 3)), found, t1 - t0, t2 - t1
        return np.concatenate(p2d), np.concatenate(p3d), found, t1 - t0, t2 - t1

    @torch.no_grad()
    def match_render(self, fq, img, depth, alpha, pose, min_alpha=0.5, lean=False, half=False):
        from mast3r.fast_nn import fast_reciprocal_NNs
        sync(self.dev)
        t0 = time.time()
        with self.amp(half):
            fr, _, _ = self.net._encode_image(img[None].clamp(0.0, 1.0) * 2.0 - 1.0, self.shape)
        sync(self.dev)
        t1 = time.time()
        with self.amp(half):
            d1, d2 = self.net._decoder(fq, self.pos, fr.to(fq.dtype), self.pos)
        desc1, desc2 = self.descriptors(d1, d2, self.shape, lean)
        sync(self.dev)
        t2 = time.time()
        xm, xq = fast_reciprocal_NNs(desc2[0], desc1[0], subsample_or_initxy1=self.subsample, ret_xy=True,
                                     device=self.dev, dist="dot", block_size=2 ** 13)
        found = len(xm)
        p2d, p3d = np.zeros((0, 2)), np.zeros((0, 3))
        if found:
            xm = torch.as_tensor(np.ascontiguousarray(xm), device=self.dev, dtype=torch.long)
            z = depth[xm[:, 1], xm[:, 0]].float()
            ok = (alpha[xm[:, 1], xm[:, 0]] > min_alpha) & torch.isfinite(z) & (z > 1e-6)
            c = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=self.dev)
            cam = torch.stack([(xm[:, 0].float() - float(self.K[0, 2])) / self.focal * z, (xm[:, 1].float() - float(self.K[1, 2])) / self.focal * z, z], -1)
            p3d = (cam @ c[:3, :3].T + c[:3, 3])[ok].double().cpu().numpy()
            p2d = np.ascontiguousarray(xq, dtype=np.float64)[ok.cpu().numpy()]
        sync(self.dev)
        return p2d, p3d, int(found), dict(encoder=t1 - t0, decoder=t2 - t1, matching=time.time() - t2)

    def localize(self, img, prior=None, idx=None, topk=1, lean=False, half=False, fq=None):
        t = {}
        sync(self.dev)
        a = time.time()
        if fq is None:
            fq = self.encode(img, half=half)
        sync(self.dev)
        t["encoder"] = time.time() - a
        a = time.time()
        if idx is None:
            if prior is not None:
                idx = self.near(prior, topk)
            else:
                idx, _ = self.retrieve(fq, topk)
                sync(self.dev)
        t["select"] = time.time() - a
        p2d, p3d, found, t["decoder"], t["matching"] = self.match(fq, idx, lean=lean, half=half)
        a = time.time()
        start = np.asarray(prior, dtype=np.float64) if prior is not None else self.poses[idx[0]]
        sol = G.solve_pnp(p3d, p2d, self.K, self.reproj, start) if len(p2d) >= 6 else None
        t["pnp"] = time.time() - a
        ok = sol is not None and bool(np.isfinite(sol["c2w"]).all())
        return dict(ok=ok, pose=sol["c2w"] if ok else start, start=start, idx=list(idx), matches=int(len(p2d)), reciprocal=int(found),
                    inliers=int(sol["inliers"]) if ok else 0, solver=sol["solver"] if ok else "none", stages=t, seconds=float(sum(t.values())))
