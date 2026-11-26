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


def robust_pnp(p3d, p2d, K, thr, init):
    from scipy.optimize import least_squares
    x = _to_x(init)
    r = least_squares(lambda v: _reproj(v, p3d, p2d, K).ravel(), x, loss="cauchy", f_scale=float(thr), max_nfev=60)
    x = r.x
    for _ in range(2):
        inl = _errors(x, p3d, p2d, K) < thr
        if int(inl.sum()) < 6:
            return None
        x = least_squares(lambda v: _reproj(v, p3d[inl], p2d[inl], K).ravel(), x, max_nfev=40).x
    inl = _errors(x, p3d, p2d, K) < thr
    return dict(c2w=_to_c2w(x), inliers=int(inl.sum()), solver="least squares from the start pose")


def solve_pnp(p3d, p2d, K, thr, init=None, iters=2000):
    p3d = np.ascontiguousarray(p3d, dtype=np.float64)
    p2d = np.ascontiguousarray(p2d, dtype=np.float64)
    if len(p3d) < 6:
        return None
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
                    return dict(c2w=_to_c2w(x), inliers=int((_errors(x, p3d, p2d, K) < thr).sum()), solver="cv2 RANSAC SQPNP")
        except Exception:
            pass
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
