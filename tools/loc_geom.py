import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

_DIS = None


def rot(v):
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if th < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(th) / th * k + (1.0 - np.cos(th)) / (th * th) * k @ k


def rotvec(r):
    r = np.asarray(r, dtype=np.float64)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    c = np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-7:
        return 0.5 * w
    if np.pi - th < 1e-5:
        d = np.sqrt(np.clip(0.5 * (np.diag(r) + 1.0), 0.0, None))
        i = int(np.argmax(d))
        ax = (r[:, i] + np.eye(3)[i]) / (2.0 * max(d[i], 1e-12))
        return th * ax / max(np.linalg.norm(ax), 1e-12)
    return th / (2.0 * np.sin(th)) * w


def intrinsics(focal, pp):
    return np.array([[float(focal), 0.0, float(pp[0]) - 0.5], [0.0, float(focal), float(pp[1]) - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def project(K, c2w, pts):
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    x = np.asarray(pts, dtype=np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = x[:, 2]
    zs = np.where(np.abs(z) > 1e-9, z, 1e-9)
    uv = x[:, :2] / zs[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def inside(uv, hw):
    return (uv[:, 0] >= -0.5) & (uv[:, 0] <= hw[1] - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= hw[0] - 0.5)


def pose_error(a, b, depth):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    c = 0.5 * (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0)
    return 100.0 * dt / max(float(depth), 1e-9), float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pixel_shift(K, ref, est, pts, hw):
    ua, za = project(K, ref, pts)
    ub, zb = project(K, est, pts)
    ok = (za > 1e-6) & inside(ua, hw)
    if int(ok.sum()) < 8:
        return float("nan")
    d = np.linalg.norm(ua[ok] - ub[ok], axis=1)
    d[zb[ok] <= 1e-6] = float(np.hypot(hw[0], hw[1]))
    return float(np.median(d))


def overlap(K, c2w, pts, hw):
    uv, z = project(K, c2w, pts)
    return float(((z > 1e-6) & inside(uv, hw)).mean()) if len(pts) else 0.0


def perturb(c2w, trans, rot_deg, rng):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    m = np.eye(4)
    m[:3, :3] = rot(np.radians(float(rot_deg)) * ax)
    m[:3, 3] = float(trans) * d
    return np.asarray(c2w, dtype=np.float64) @ m


def similarity(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ma, mb = a.mean(0), b.mean(0)
    xa, xb = a - ma, b - mb
    U, D, Vt = np.linalg.svd(xb.T @ xa / len(a))
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U) * np.linalg.det(Vt))) or 1.0])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max(float((xa ** 2).sum()) / len(a), 1e-18))
    return s, R, mb - s * R @ ma


def move_pose(s, R, t, pose):
    pose = np.asarray(pose, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = R @ pose[:3, :3]
    out[:3, 3] = s * R @ pose[:3, 3] + t
    return out


def fit_poses(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ca, cb = a[:, :3, 3], b[:, :3, 3]
    ra = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    rb = float(np.sqrt(((cb - cb.mean(0)) ** 2).sum(1).mean()))
    if ra < 1e-12 or rb < 1e-12:
        return None
    pa = np.concatenate([ca] + [ca + ra * a[:, :3, k] for k in range(3)])
    pb = np.concatenate([cb] + [cb + rb * b[:, :3, k] for k in range(3)])
    return similarity(pa, pb)


def transfer_poses(ref_names, ref_poses, map_names, map_poses, want):
    ref_names = list(ref_names)
    pairs = [(i, ref_names.index(n)) for i, n in enumerate(map_names) if n in ref_names]
    if len(pairs) < 3:
        return None
    A = np.asarray(ref_poses, dtype=np.float64)[[j for _, j in pairs]]
    Bm = np.asarray(map_poses, dtype=np.float64)[[i for i, _ in pairs]]
    fit = fit_poses(A, Bm)
    if fit is None:
        return None
    s, R, t = fit
    moved = [move_pose(s, R, t, p) for p in A]
    et = [float(np.linalg.norm(m[:3, 3] - q[:3, 3])) for m, q in zip(moved, Bm)]
    er = [float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(m[:3, :3].T @ q[:3, :3]) - 1.0), -1.0, 1.0)))) for m, q in zip(moved, Bm)]
    out = {n: move_pose(s, R, t, np.asarray(ref_poses, dtype=np.float64)[ref_names.index(n)]) for n in want if n in ref_names}
    return out, dict(n=len(pairs), scale=float(s), idx=[i for i, _ in pairs], et=et, er=er)


def motion(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    c = 0.5 * (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def quat_rot(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                     [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                     [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]])


def _colmap_pose(q, t):
    w2c = np.eye(4)
    w2c[:3, :3] = quat_rot(q)
    w2c[:3, 3] = np.asarray(t, dtype=np.float64)
    return np.linalg.inv(w2c)


def read_poses(path):
    import json
    import os
    import struct
    if os.path.isdir(path):
        for sub in ("", "sparse/0", "sparse", "0"):
            for f in ("images.bin", "images.txt"):
                q = os.path.join(path, sub, f)
                if os.path.isfile(q):
                    return read_poses(q)
        raise FileNotFoundError("no images.bin or images.txt under %s" % path)
    names, poses = [], []
    if path.endswith(".json"):
        d = json.load(open(path))
        items = d.items() if isinstance(d, dict) else [(x["name"], x) for x in d]
        for n, v in items:
            m = np.asarray(v["c2w"] if isinstance(v, dict) else v, dtype=np.float64)
            if m.shape == (4, 4):
                names.append(os.path.basename(str(n)))
                poses.append(m)
    elif path.endswith(".bin"):
        with open(path, "rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                v = struct.unpack("<I7dI", f.read(64))
                name = b""
                while True:
                    ch = f.read(1)
                    if ch in (b"\x00", b""):
                        break
                    name += ch
                f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
                names.append(os.path.basename(name.decode("utf-8")))
                poses.append(_colmap_pose(v[1:5], v[5:8]))
    else:
        rows = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        for l in rows:
            p = l.split()
            if len(p) >= 10 and p[9].lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    v = [float(x) for x in p[1:8]]
                except ValueError:
                    continue
                names.append(os.path.basename(" ".join(p[9:])))
                poses.append(_colmap_pose(v[:4], v[4:7]))
    return names, np.asarray(poses, dtype=np.float64)


def _to_x(c2w):
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    return np.r_[rotvec(w2c[:3, :3]), w2c[:3, 3]]


def _to_c2w(x):
    w2c = np.eye(4)
    w2c[:3, :3] = rot(x[:3])
    w2c[:3, 3] = x[3:]
    return np.linalg.inv(w2c)


def _reproj(x, p3d, p2d, K):
    c = p3d @ rot(x[:3]).T + x[3:]
    z = np.where(np.abs(c[:, 2]) > 1e-9, c[:, 2], 1e-9)
    return c[:, :2] / z[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2] - p2d


def _errors(x, p3d, p2d, K):
    c = p3d @ rot(x[:3]).T + x[3:]
    e = np.linalg.norm(_reproj(x, p3d, p2d, K), axis=1)
    e[c[:, 2] <= 1e-9] = np.inf
    return e


def _skew(c):
    z = np.zeros(len(c))
    return np.stack([np.stack([z, -c[:, 2], c[:, 1]], 1), np.stack([c[:, 2], z, -c[:, 0]], 1), np.stack([-c[:, 1], c[:, 0], z], 1)], 1)


def _gauss_newton(x, p3d, p2d, K, weight, steps):
    R, t = rot(x[:3]), np.array(x[3:], dtype=np.float64)
    for _ in range(steps):
        c = p3d @ R.T + t
        z = np.where(np.abs(c[:, 2]) > 1e-9, c[:, 2], 1e-9)
        r = np.stack([K[0, 0] * c[:, 0] / z + K[0, 2] - p2d[:, 0], K[1, 1] * c[:, 1] / z + K[1, 2] - p2d[:, 1]], 1)
        w = weight(r, c)
        Jc = np.zeros((len(c), 2, 3))
        Jc[:, 0, 0] = K[0, 0] / z
        Jc[:, 0, 2] = -K[0, 0] * c[:, 0] / (z * z)
        Jc[:, 1, 1] = K[1, 1] / z
        Jc[:, 1, 2] = -K[1, 1] * c[:, 1] / (z * z)
        J = np.concatenate([-Jc @ _skew(c - t), Jc], 2)
        Jw = J * w[:, None, None]
        Hm = np.einsum("nij,nik->jk", Jw, J)
        d = np.linalg.solve(Hm + 1e-9 * np.trace(Hm) * np.eye(6), -np.einsum("nij,ni->j", Jw, r))
        R, t = rot(d[:3]) @ R, t + d[3:]
    return np.r_[rotvec(R), t]


def polish(x, p3d, p2d, K, thr, rounds=3):
    s2 = float(thr) ** 2
    x = _gauss_newton(x, p3d, p2d, K, lambda r, c: (c[:, 2] > 1e-9) / (1.0 + (r * r).sum(1) / s2), 6)
    for _ in range(rounds):
        inl = _errors(x, p3d, p2d, K) < thr
        if int(inl.sum()) < 6:
            return None
        x = _gauss_newton(x, p3d[inl], p2d[inl], K, lambda r, c: np.ones(len(r)), 3)
    return x if np.isfinite(x).all() else None


def robust_pnp(p3d, p2d, K, thr, init):
    from scipy.optimize import least_squares
    x = least_squares(lambda v: _reproj(v, p3d, p2d, K).ravel(), _to_x(init), loss="cauchy", f_scale=float(thr), max_nfev=60).x
    x = polish(x, p3d, p2d, K, thr)
    if x is None:
        return None
    return dict(c2w=_to_c2w(x), inliers=int((_errors(x, p3d, p2d, K) < thr).sum()), solver="least squares from the start pose")


def solve_pnp(p3d, p2d, K, thr, init=None, iters=2000):
    p3d = np.ascontiguousarray(p3d, dtype=np.float64)
    p2d = np.ascontiguousarray(p2d, dtype=np.float64)
    if len(p3d) < 6:
        return None
    cands = []
    if cv2 is not None:
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(p3d, p2d, K, None, iterationsCount=int(iters), reprojectionError=float(thr),
                                                     confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
            if ok and inl is not None and len(inl) >= 6:
                inl = inl.ravel()
                if hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(p3d[inl], p2d[inl], K, None, rvec, tvec)
                x = np.r_[np.asarray(rvec, dtype=np.float64).ravel(), np.asarray(tvec, dtype=np.float64).ravel()]
                if np.isfinite(x).all():
                    y = polish(x, p3d, p2d, K, thr)
                    cands.append((x if y is None else y, "cv2 RANSAC SQPNP, refit on all matches"))
        except Exception:
            pass
    if init is not None and cands:
        try:
            y = polish(_to_x(init), p3d, p2d, K, thr)
            if y is not None:
                cands.append((y, "start pose, refit on all matches"))
        except Exception:
            pass
    if cands:
        x, name = max(cands, key=lambda c: int((_errors(c[0], p3d, p2d, K) < thr).sum()))
        return dict(c2w=_to_c2w(x), inliers=int((_errors(x, p3d, p2d, K) < thr).sum()), solver=name)
    if init is None:
        return None
    try:
        return robust_pnp(p3d, p2d, K, thr, init)
    except Exception:
        return None


def flow_residual(real, rend, mask=None, border=8):
    global _DIS
    if cv2 is None or not hasattr(cv2, "DISOpticalFlow_create"):
        return float("nan")
    try:
        def grey(x):
            u = np.ascontiguousarray((np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8))
            return cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

        if _DIS is None:
            _DIS = (cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM), cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)))
        dis, clahe = _DIS
        a, b = clahe.apply(grey(real)), clahe.apply(grey(rend))
        mag = np.linalg.norm(dis.calc(a, b, None), axis=-1)
        sel = np.zeros(mag.shape, dtype=bool)
        sel[border:-border, border:-border] = True
        if mask is not None:
            sel &= np.asarray(mask, dtype=bool)
        if int(sel.sum()) < 200:
            return float("nan")
        return float(np.median(mag[sel]))
    except Exception:
        return float("nan")


def covisible(K, ref, other, pts, hw):
    uv, z = project(K, ref, pts)
    a = (z > 1e-6) & inside(uv, hw)
    if int(a.sum()) < 8:
        return 0.0
    uv2, z2 = project(K, other, np.asarray(pts)[a])
    return float(((z2 > 1e-6) & inside(uv2, hw)).mean())
