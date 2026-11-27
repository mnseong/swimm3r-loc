import argparse
import ast
import csv
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zlib

import numpy as np

BASE = "https://data.archive.sigma2.no/dataset/24978c88-334f-4024-b726-fe64e8704046/download/"
META = ("camera_poses.csv", "camera_footprints.csv", "camera_calibrations.csv", "camera_sensors.csv", "camera_groups.csv")
SURVEYS = {"qdch0ftq": {"2010": "20100428_020202", "2011": "20110415_020103", "2012": "20120430_002423", "2013": "20130406_023610"}}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch(url, rng=None, tries=5):
    for k in range(tries):
        try:
            hd = {"User-Agent": "sealoc-scene"}
            if rng is not None:
                hd["Range"] = "bytes=%s" % rng
            with urllib.request.urlopen(urllib.request.Request(url, headers=hd), timeout=180) as r:
                return r.read(), r.headers.get("Content-Range")
        except Exception as e:
            if k == tries - 1:
                raise
            print("[sealoc] %s failed (%s), again in %d s" % (url.rsplit("/", 1)[-1], e, 3 * (k + 1)), flush=True)
            time.sleep(3 * (k + 1))


def fetch_range(url, a, b):
    d, _ = fetch(url, "%d-%d" % (a, b - 1))
    if len(d) != b - a:
        raise IOError("range %d-%d of %s gave %d bytes" % (a, b, url, len(d)))
    return d


def meta(dl):
    os.makedirs(os.path.join(dl, "meta"), exist_ok=True)
    for f in META:
        p = os.path.join(dl, "meta", f)
        if not os.path.isfile(p):
            print("[sealoc] downloading %s" % f, flush=True)
            d, _ = fetch(BASE + f)
            open(p + ".part", "wb").write(d)
            os.replace(p + ".part", p)
    return os.path.join(dl, "meta")


def central_directory(url, cache):
    if os.path.isfile(cache):
        return json.load(open(cache))
    tail, cr = fetch(url, "-65536")
    total = int(cr.rsplit("/", 1)[-1])
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("no end of central directory in %s" % url)
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", tail[i:i + 22])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x07", 0, i + 1)
        off64 = struct.unpack("<4sLQL", tail[j:j + 20])[2]
        rec = fetch_range(url, off64, off64 + 56)
        _, _, _, _, _, _, _, n, cd_size, cd_off = struct.unpack("<4sQ2H2L4Q", rec)
    cd = fetch_range(url, cd_off, cd_off + cd_size)
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        h = struct.unpack("<4s6H3L5H2L", cd[p:p + 46])
        method, csize, usize, nl, xl, cl, lho = h[4], h[8], h[9], h[10], h[11], h[12], h[16]
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        ex = cd[p + 46 + nl:p + 46 + nl + xl]
        q = 0
        while q + 4 <= len(ex):
            tag, sz = struct.unpack("<2H", ex[q:q + 4])
            if tag == 1:
                vals, r = [], q + 4
                for want in (usize, csize, lho):
                    if want == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", ex[r:r + 8])[0])
                        r += 8
                    else:
                        vals.append(want)
                usize, csize, lho = vals
            q += 4 + sz
        out[os.path.basename(name)] = dict(name=name, method=method, csize=csize, usize=usize, lho=lho, crc=h[7])
        p += 46 + nl + xl + cl
    json.dump(dict(total=total, members=out), open(cache, "w"))
    return dict(total=total, members=out)


def member(url, m):
    hd = fetch_range(url, m["lho"], m["lho"] + 30)
    nl, xl = struct.unpack("<2H", hd[26:30])
    a = m["lho"] + 30 + nl + xl
    raw = fetch_range(url, a, a + m["csize"])
    if m["method"] == 8:
        raw = zlib.decompressobj(-15).decompress(raw)
    elif m["method"] != 0:
        raise IOError("compression method %d of %s" % (m["method"], m["name"]))
    if (zlib.crc32(raw) & 0xFFFFFFFF) != m["crc"]:
        raise IOError("crc of %s does not match" % m["name"])
    return raw


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def ecef(lon, lat, h):
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lo, la = np.radians(lon), np.radians(lat)
    n = a / np.sqrt(1.0 - e2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1.0 - e2) + h) * np.sin(la)], -1)


class Enu:
    def __init__(self, lon, lat, h):
        self.o = ecef(np.float64(lon), np.float64(lat), np.float64(h))
        lo, la = math.radians(lon), math.radians(lat)
        self.R = np.array([[-math.sin(lo), math.cos(lo), 0.0], [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                           [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])

    def __call__(self, lon, lat, h):
        return (ecef(np.asarray(lon, np.float64), np.asarray(lat, np.float64), np.asarray(h, np.float64)) - self.o) @ self.R.T


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


C0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def conventions():
    out = []
    for world in (True, False):
        for sy in (1, -1):
            for off in (0, 90, 180, 270):
                for swap in (False, True):
                    for sp in (1, -1):
                        for sr in (1, -1):
                            out.append(dict(world=world, sy=sy, off=off, swap=swap, sp=sp, sr=sr))
    return out


def rotation(yaw, pitch, roll, cv):
    p, r = math.radians(cv["sp"] * pitch), math.radians(cv["sr"] * roll)
    A = ry(r) @ rx(p) if cv["swap"] else rx(p) @ ry(r)
    Z = rz(math.radians(cv["sy"] * yaw + cv["off"]))
    return (A @ Z if cv["world"] else Z @ A) @ C0


def distort(xn, cal):
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    rad = 1.0 + cal["k1"] * r2 + cal["k2"] * r2 ** 2 + cal["k3"] * r2 ** 3
    xd = x * rad + 2.0 * cal["p1"] * x * y + cal["p2"] * (r2 + 2.0 * x * x)
    yd = y * rad + cal["p1"] * (r2 + 2.0 * y * y) + 2.0 * cal["p2"] * x * y
    return np.stack([xd, yd], -1)


def undistort(uv, cal):
    xd = (np.asarray(uv, np.float64) - [cal["cx"], cal["cy"]]) / [cal["fx"], cal["fy"]]
    x = xd.copy()
    for _ in range(50):
        x = x + (xd - distort(x, cal))
    return x


def corner_rays(cal):
    cs = np.array([[0.0, 0.0], [cal["width"], 0.0], [cal["width"], cal["height"]], [0.0, cal["height"]]])
    d = np.c_[undistort(cs, cal), np.ones(4)]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def corner_error(c2w, pts, rays):
    dw = pts - c2w[:3, 3]
    dw = dw / np.linalg.norm(dw, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(((c2w[:3, :3] @ rays.T).T * dw).sum(1), -1.0, 1.0))).mean())


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ccw(p):
    x, y = p[:, 0], p[:, 1]
    return p if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0 else p[::-1]


def clip(subj, clipper):
    out = list(subj)
    n = len(clipper)
    for i in range(n):
        a, b = clipper[i], clipper[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = inp[-1]
        for e in inp:
            if side(e) >= 0:
                if side(s) < 0:
                    out.append(cross(s, e, a, b))
                out.append(e)
            elif side(s) >= 0:
                out.append(cross(s, e, a, b))
            s = e
    return np.array(out) if len(out) >= 3 else None


def cross(p, q, a, b):
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-15:
        return q
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


def iou(p, q):
    if np.linalg.norm(p.mean(0) - q.mean(0)) > 0.5 * (np.ptp(p, 0).max() + np.ptp(q, 0).max()):
        return 0.0
    c = clip(p, q)
    if c is None:
        return 0.0
    i = area(c)
    return i / max(area(p) + area(q) - i, 1e-12)


def corners(text):
    return np.array(ast.literal_eval(text), dtype=np.float64)


def survey_rows(poses, prints, survey):
    day = survey.split("_")[0]
    rows = [r for r in poses if r["camera_label"].startswith("PR_" + day) and r["camera_label"].endswith("_LC16")]
    fp = {r["camera_label"]: r for r in prints}
    rows = [r for r in rows if r["camera_label"] in fp]
    rows.sort(key=lambda r: r["timestamp"])
    return rows, fp


def ts(r):
    import datetime
    return datetime.datetime.fromisoformat(r["timestamp"]).timestamp()


def heading_gap(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def select(args, poses, prints):
    mrows, fp = survey_rows(poses, prints, args.map_survey)
    qrows, _ = survey_rows(poses, prints, args.query_survey)
    if len(mrows) < args.n_map:
        raise SystemExit("the map survey has %d left images with a pose and a footprint" % len(mrows))
    lon0 = float(np.mean([float(r["longitude"]) for r in mrows]))
    lat0 = float(np.mean([float(r["latitude"]) for r in mrows]))
    h0 = float(np.mean([float(r["height"]) for r in mrows]))
    enu = Enu(lon0, lat0, h0)

    def poly(r):
        c = corners(fp[r["camera_label"]]["coordinates"])
        return enu(c[:, 0], c[:, 1], c[:, 2])

    mp = [ccw(poly(r)[:4, :2]) for r in mrows]
    qp = [ccw(poly(r)[:4, :2]) for r in qrows]
    mc = np.array([p.mean(0) for p in mp])
    qc = np.array([p.mean(0) for p in qp])
    t = np.array([ts(r) for r in mrows])
    ok_start = [i for i in range(len(mrows) - args.n_map + 1) if float(np.diff(t[i:i + args.n_map]).max()) <= args.max_gap]
    if not ok_start:
        raise SystemExit("no run of %d map images without a gap above %.1f s" % (args.n_map, args.max_gap))
    reach = 0.5 * float(np.median([np.ptp(p, 0).max() for p in mp])) * 2.0
    best = None
    for i in ok_start[::max(1, args.n_map // 10)]:
        lo, hi = mc[i:i + args.n_map].min(0) - reach, mc[i:i + args.n_map].max(0) + reach
        cand = [k for k in range(len(qrows)) if (qc[k] >= lo).all() and (qc[k] <= hi).all()]
        n_ok = 0
        for k in cand:
            d = np.linalg.norm(mc[i:i + args.n_map] - qc[k], axis=1)
            near = np.argsort(d)[:6] + i
            if max(iou(qp[k], mp[j]) for j in near) >= args.min_iou:
                n_ok += 1
        if best is None or n_ok > best[1]:
            best = (i, n_ok)
    i0 = best[0]
    lo = max(0, i0 - args.n_map // 10)
    hi = min(ok_start[-1], i0 + args.n_map // 10)
    for i in [s for s in ok_start if lo <= s <= hi]:
        if i == best[0]:
            continue
        lo_, hi_ = mc[i:i + args.n_map].min(0) - reach, mc[i:i + args.n_map].max(0) + reach
        n_ok = 0
        for k in range(len(qrows)):
            if not ((qc[k] >= lo_).all() and (qc[k] <= hi_).all()):
                continue
            d = np.linalg.norm(mc[i:i + args.n_map] - qc[k], axis=1)
            if max(iou(qp[k], mp[j]) for j in np.argsort(d)[:6] + i) >= args.min_iou:
                n_ok += 1
        if n_ok > best[1]:
            best = (i, n_ok)
    i0 = best[0]
    msel = list(range(i0, i0 + args.n_map))
    info = []
    for k in range(len(qrows)):
        d = np.linalg.norm(mc[i0:i0 + args.n_map] - qc[k], axis=1)
        near = np.argsort(d)[:8] + i0
        ious = [iou(qp[k], mp[j]) for j in near]
        j = int(near[int(np.argmax(ious))])
        if max(ious) >= args.min_iou:
            yq, ym = float(qrows[k]["yaw"]), float(mrows[j]["yaw"])
            jn = int(np.argmin(d)) + i0
            info.append(dict(k=k, iou=float(max(ious)), best=j, nearest=jn, dist=float(d.min()),
                             heading_gap=heading_gap(yq, ym), heading_gap_nearest=heading_gap(yq, float(mrows[jn]["yaw"]))))
    if len(info) < args.n_query:
        print("[sealoc] only %d images of the query survey overlap the map frames (footprint IoU at least %.2f), all are taken" % (len(info), args.min_iou),
              flush=True)
        qsel = info
    else:
        pick = np.unique(np.round(np.linspace(0, len(info) - 1, args.n_query)).astype(int))
        qsel = [info[int(x)] for x in pick]
    alt = [float(enu(float(mrows[i]["longitude"]), float(mrows[i]["latitude"]), float(mrows[i]["height"]))[2] - poly(mrows[i])[:4, 2].mean()) for i in msel]
    return dict(enu=enu, origin=[lon0, lat0, h0], mrows=mrows, qrows=qrows, msel=msel, qsel=qsel, candidates=len(info), altitude=float(np.median(alt)))


def calibrations(dl, site, surveys):
    cal = {r["calibration_id"]: {k: float(v) for k, v in r.items() if k != "calibration_id"} for r in read_csv(os.path.join(dl, "meta", "camera_calibrations.csv"))}
    sen = read_csv(os.path.join(dl, "meta", "camera_sensors.csv"))
    grp = {r["camera_group_label"]: r["camera_group_id"] for r in read_csv(os.path.join(dl, "meta", "camera_groups.csv"))}
    out = {}
    for sv in surveys:
        gid = grp.get("%s_%s" % (site, sv))
        left = [x for x in sen if x["camera_group_id"] == gid and x["sensor_label"] == "stereo_left" and not x["master_sensor_id"].strip()]
        if gid is None or len(left) != 1 or left[0]["sensor_calibration_id"] not in cal:
            raise SystemExit("no single left camera with a calibration for survey %s_%s" % (site, sv))
        cid = left[0]["sensor_calibration_id"]
        out[sv] = (cal[cid], cid)
    return out


def pose_c2w(r, enu, cv):
    c = enu(float(r["longitude"]), float(r["latitude"]), float(r["height"]))
    m = np.eye(4)
    m[:3, :3] = rotation(float(r["yaw"]), float(r["pitch"]), float(r["roll"]), cv)
    m[:3, 3] = c
    return m


def pick_convention(rows, fp, enu, cal_of, n=80):
    sel = rows[::max(1, len(rows) // n)][:n]
    rays = [corner_rays(cal_of(r)) for r in sel]
    pts = []
    for r in sel:
        c = corners(fp[r["camera_label"]]["coordinates"])[:4]
        pts.append(enu(c[:, 0], c[:, 1], c[:, 2]))
    res = []
    for cv in conventions():
        e = [corner_error(pose_c2w(r, enu, cv), p, ry) for r, p, ry in zip(sel, pts, rays)]
        res.append((float(np.median(e)), float(np.percentile(e, 90)), cv))
    res.sort(key=lambda x: x[0])
    return res[0][2], res[0][0], res[0][1], res[1][0]


def decode(raw, name, source):
    import cv2
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError("%s could not be decoded" % name)
    if a.ndim == 2:
        raise IOError("%s is one channel (%s), a demosaic pattern is needed" % (name, a.dtype))
    a = a[..., :3][..., ::-1]
    full = 65535.0 if a.dtype == np.uint16 else 255.0
    return a.astype(np.float32) / full


def camera(cal):
    K0 = np.array([[cal["fx"], 0.0, cal["cx"]], [0.0, cal["fy"], cal["cy"]], [0.0, 0.0, 1.0]])
    return K0, np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"], cal["k3"]])


def covers(K0, dist, K1, W, H):
    import cv2
    mx, my = cv2.initUndistortRectifyMap(K0, dist, None, K1, (W, H), cv2.CV_32FC1)
    return bool(mx.min() >= 0.0 and mx.max() <= W - 1.0 and my.min() >= 0.0 and my.max() <= H - 1.0)


def geometry(cals, width):
    import cv2
    cs = list(cals.values())
    W, H = int(cs[0]["width"]), int(cs[0]["height"])
    if any((int(c["width"]), int(c["height"])) != (W, H) for c in cs):
        raise SystemExit("the calibrations differ in image size")
    src = {k: camera(c) for k, c in cals.items()}
    opt = [cv2.getOptimalNewCameraMatrix(K0, d, (W, H), 0.0, (W, H))[0] for K0, d in src.values()]
    K1 = np.array([[max(k[0, 0] for k in opt), 0.0, float(np.mean([k[0, 2] for k in opt]))],
                   [0.0, max(k[1, 1] for k in opt), float(np.mean([k[1, 2] for k in opt]))], [0.0, 0.0, 1.0]])
    for _ in range(200):
        if all(covers(K0, d, K1, W, H) for K0, d in src.values()):
            break
        K1[0, 0] *= 1.002
        K1[1, 1] *= 1.002
    else:
        raise SystemExit("no common undistorted camera without an empty border")
    hc = min(H, int(round(W * 3 / 4)))
    top = (H - hc) // 2
    s = width / float(W)
    hn = int(round(hc * s)) // 16 * 16
    s_y = hn / float(hc)
    K = np.array([[K1[0, 0] * s, 0.0, (K1[0, 2] + 0.5) * s - 0.5], [0.0, K1[1, 1] * s_y, (K1[1, 2] - top + 0.5) * s_y - 0.5], [0.0, 0.0, 1.0]])
    return dict(top=top, hc=hc, w=width, h=hn, K=K, K1=K1, src=src, zoom=float(K1[0, 0] / max(k[0, 0] for k in opt)))


def prepare(lin, geo, key):
    import cv2
    K0, dist = geo["src"][key]
    u = cv2.undistort(lin, K0, dist, None, geo["K1"])
    u = u[geo["top"]:geo["top"] + geo["hc"]]
    return cv2.resize(u, (geo["w"], geo["h"]), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True, help="folder for the downloads (metadata, central directories, raw images), outside the repository")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "sealoc_s1"))
    ap.add_argument("--site", default="qdch0ftq")
    ap.add_argument("--map_year", default="2011")
    ap.add_argument("--query_year", default="2012")
    ap.add_argument("--n_map", type=int, default=90)
    ap.add_argument("--n_query", type=int, default=24)
    ap.add_argument("--min_iou", type=float, default=0.2)
    ap.add_argument("--max_gap", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--source", default="raw", choices=("raw", "grayworld"))
    ap.add_argument("--white", type=float, default=99.5, help="percentile of all pixels of all chosen images that maps to white, one value for all")
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--query_base", type=int, default=100000)
    ap.add_argument("--select_only", action="store_true")
    args = ap.parse_args()
    dl = os.path.abspath(args.dl)
    if os.path.commonpath([dl, HERE]) == HERE:
        raise SystemExit("--dl must lie outside the repository")
    args.map_survey = SURVEYS[args.site][args.map_year]
    args.query_survey = SURVEYS[args.site][args.query_year]
    meta(dl)
    poses = read_csv(os.path.join(dl, "meta", "camera_poses.csv"))
    prints = read_csv(os.path.join(dl, "meta", "camera_footprints.csv"))
    cals = calibrations(dl, args.site, (args.map_survey, args.query_survey))
    for sv, (c, cid) in cals.items():
        print("[sealoc] survey %s, left camera calibration %s: fx %.1f, cx %.1f, cy %.1f" % (sv, cid, c["fx"], c["cx"], c["cy"]), flush=True)
    s = select(args, poses, prints)
    mrows, qrows, enu = s["mrows"], s["qrows"], s["enu"]
    _, fp = survey_rows(poses, prints, args.map_survey)
    _, fq = survey_rows(poses, prints, args.query_survey)
    mset = set(r["camera_label"] for r in mrows)
    cal_of = lambda r: cals[args.map_survey if r["camera_label"] in mset else args.query_survey][0]
    cv, err, err90, err2 = pick_convention(mrows + qrows, dict(fp, **fq), enu, cal_of)
    print("[sealoc] orientation: tilt about the %s axes, yaw sign %d, offset %d deg, pitch sign %d, roll sign %d | the rays of the image corners miss the "
          "footprint corners by a median of %.3f deg (90 %% below %.3f deg, next convention %.3f deg)" % (
              "world" if cv["world"] else "body", cv["sy"], cv["off"], cv["sp"], cv["sr"], err, err90, err2), flush=True)
    gaps = [q["heading_gap"] for q in s["qsel"]]
    print("[sealoc] map: %d images of survey %s from %s | queries: %d of %d overlapping images of survey %s, heading gap to the best overlapping map "
          "frame median %.0f deg (below 30: %d, 30 to 150: %d, above 150: %d)" % (
              len(s["msel"]), args.map_survey, mrows[s["msel"][0]]["camera_label"], len(s["qsel"]), s["candidates"], args.query_survey,
              float(np.median(gaps)) if gaps else float("nan"), sum(g < 30 for g in gaps), sum(30 <= g <= 150 for g in gaps), sum(g > 150 for g in gaps)), flush=True)
    if args.select_only:
        return
    zips = {}
    for sv in (args.map_survey, args.query_survey):
        f = "%s_%s_images_%s.zip" % (args.site, sv, args.source)
        zips[sv] = (BASE + f, central_directory(BASE + f, os.path.join(dl, f + ".cd.json")))
    geo = geometry({sv: c for sv, (c, _) in cals.items()}, args.width)
    print("[sealoc] common undistorted camera for both surveys: fx %.1f fy %.1f cx %.1f cy %.1f at full size, %.1f %% zoom over the narrower survey" % (
        geo["K1"][0, 0], geo["K1"][1, 1], geo["K1"][0, 2], geo["K1"][1, 2], 100.0 * (geo["zoom"] - 1.0)), flush=True)
    items = [("%06d" % n, mrows[i], args.map_survey, "map") for n, i in enumerate(s["msel"])]
    items += [("%06d" % (args.query_base + n), qrows[q["k"]], args.query_survey, "query") for n, q in enumerate(s["qsel"])]
    os.makedirs(os.path.join(dl, args.source), exist_ok=True)
    lins, sat = [], []
    t0 = time.time()
    for stem, r, sv, kind in items:
        url, cd = zips[sv]
        lab = r["camera_label"]
        m = [v for k, v in cd["members"].items() if os.path.splitext(k)[0] == lab]
        if not m:
            raise SystemExit("%s is not in %s" % (lab, url))
        m = m[0]
        p = os.path.join(dl, args.source, os.path.basename(m["name"]))
        if not os.path.isfile(p):
            raw = member(url, m)
            open(p + ".part", "wb").write(raw)
            os.replace(p + ".part", p)
        x = decode(open(p, "rb").read(), p, args.source)
        sat.append(float((x >= 1.0).any(2).mean()))
        lins.append(prepare(x, geo, sv))
        print("[sealoc] %s %s <- %s (%d of %d, %.0f s)" % (kind, stem, lab, len(lins), len(items), time.time() - t0), flush=True)
    allpx = np.concatenate([x[::4, ::4].reshape(-1) for x in lins])
    black = 0.0
    white = float(np.percentile(allpx, args.white))
    import cv2
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    names = []
    for (stem, r, sv, kind), x in zip(items, lins):
        v = np.clip((x - black) / max(white - black, 1e-6), 0.0, 1.0) ** (1.0 / args.gamma)
        n = stem + ".jpg"
        cv2.imwrite(os.path.join(out, "images", n), (v[..., ::-1] * 255.0 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        names.append(n)
    mnames = names[:len(s["msel"])]
    qnames = names[len(s["msel"]):]
    ref = {}
    for (stem, r, sv, kind), n in zip(items, names):
        ref[n] = dict(c2w=pose_c2w(r, enu, cv).tolist(), label=r["camera_label"], survey=sv, kind=kind)
    json.dump(ref, open(os.path.join(out, "reference_poses.json"), "w"), indent=1)
    json.dump(dict(split_mode="cross_visit", map_survey=args.map_survey, query_survey=args.query_survey, n_train_views=len(mnames), n_test_views=len(qnames),
                   all_files=names, train_files=mnames, test_files=qnames, centre_files=[]), open(os.path.join(out, "split.json"), "w"), indent=1)
    open(os.path.join(out, "train_files.txt"), "w").write("\n".join(mnames) + "\n")
    qinfo = []
    nm = len(mnames)
    for n, q, sq in zip(qnames, s["qsel"], sat[nm:]):
        qinfo.append(dict(name=n, label=qrows[q["k"]]["camera_label"], saturated_share=sq, iou=q["iou"], best_map=mnames[q["best"] - s["msel"][0]],
                          nearest_map=mnames[q["nearest"] - s["msel"][0]], distance_m=q["dist"], heading_gap_deg=q["heading_gap"],
                          heading_gap_nearest_deg=q["heading_gap_nearest"]))
    json.dump(dict(dataset="SEALOC (Larsen and Pizarro, arXiv 2603.04056), CC BY 4.0, doi 10.11582/2026.qro1lf3z", site=args.site,
                   map_survey=args.map_survey, query_survey=args.query_survey, source=args.source,
                   tone="one global mapping for every image and channel: linear raw divided by the %.1f percentile of all pixels of all chosen images, clipped, "
                        "gamma 1/%.1f, no per image or per channel gain" % (args.white, args.gamma) if args.source == "raw" else
                        "grayworld images of the archive (per image colour normalisation by the dataset authors)",
                   white=white, gamma=args.gamma, calibration_id={sv: cid for sv, (_, cid) in cals.items()}, calibration={sv: c for sv, (c, _) in cals.items()},
                   undistorted="each survey with its own left camera calibration into one common camera", undistorted_K_full=geo["K1"].tolist(),
                   crop_rows=[geo["top"], geo["top"] + geo["hc"]],
                   K=geo["K"].tolist(), hw=[geo["h"], geo["w"]], enu_origin_lon_lat_h=s["origin"], frame="local east north up in metres, OpenCV camera axes, c2w",
                   orientation=dict(cv, corner_deg=err, corner_deg_p90=err90, next_deg=err2), min_iou=args.min_iou, candidates=s["candidates"],
                   altitude_m_median=s["altitude"],
                   map_labels=[mrows[i]["camera_label"] for i in s["msel"]], map_saturated_share=sat[:nm],
                   saturated_share_median=dict(map=float(np.median(sat[:nm])), queries=float(np.median(sat[nm:])) if sat[nm:] else None,
                                               map_max=float(max(sat[:nm])), queries_max=float(max(sat[nm:])) if sat[nm:] else None), queries=qinfo),
              open(os.path.join(out, "scene.json"), "w"), indent=1)
    mb = sum(os.path.getsize(os.path.join(out, "images", n)) for n in names) / 1e6
    print("[sealoc] wrote %d map frames and %d queries to %s (%.1f MB of images), white point %.4f of the raw range, median height of the map cameras "
          "above their footprints %.2f m" % (len(mnames), len(qnames), out, mb, white, s["altitude"]), flush=True)
    print("[sealoc] pixels at the top of the raw range in any channel: map frames median %.1f %% (largest %.1f %%), queries median %.1f %% (largest %.1f %%)" % (
        100.0 * np.median(sat[:nm]), 100.0 * max(sat[:nm]), 100.0 * np.median(sat[nm:]) if sat[nm:] else 0.0, 100.0 * max(sat[nm:]) if sat[nm:] else 0.0),
          flush=True)


if __name__ == "__main__":
    main()
