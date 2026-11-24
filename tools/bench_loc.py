import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

import eval_gaussian_map as E

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def parse_tegrastats(line):
    out = {}
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        out["ram_used_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    mw = {}
    for name, cur in re.findall(r"([A-Z][A-Z0-9_]+) (\d+)mW/\d+mW", line):
        if name != "NC":
            mw[name] = mw.get(name, 0) + int(cur)
    if mw:
        out["mw"] = mw
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        out["gpu_util"] = int(m.group(1))
    return out


class Tegra:
    def __init__(self, interval_ms=500):
        self.rows, self.proc = [], None
        exe = shutil.which("tegrastats")
        if exe is None:
            return
        try:
            self.proc = subprocess.Popen([exe, "--interval", str(interval_ms)], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception:
            self.proc = None

    def _read(self):
        try:
            for line in self.proc.stdout:
                r = parse_tegrastats(line)
                if r:
                    self.rows.append((time.time(), r))
        except Exception:
            pass

    def window(self, t0, t1):
        return [r for t, r in list(self.rows) if t0 <= t <= t1]

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


def power_summary(rows):
    rows = [r for r in rows if "mw" in r]
    if not rows:
        return None
    names = sorted({k for r in rows for k in r["mw"]})
    mean = {k: float(np.mean([r["mw"].get(k, 0) for r in rows])) / 1000.0 for k in names}
    used = [k for k in RAILS if k in mean] if all(k in mean for k in RAILS) else names
    return {"rails_w": mean, "sum_of": used, "total_w": float(sum(mean[k] for k in used)), "samples": len(rows)}


def predict_pose(known, num):
    from scipy.spatial.transform import Rotation
    prev = sorted(k for k in known if k < num)
    if len(prev) >= 2:
        a, b = prev[-2], prev[-1]
        ta, tb = known[a], known[b]
        s = (num - b) / float(b - a)
        rel = Rotation.from_matrix(ta[:3, :3].T @ tb[:3, :3]).as_rotvec()
        out = np.eye(4)
        out[:3, :3] = tb[:3, :3] @ Rotation.from_rotvec(s * rel).as_matrix()
        out[:3, 3] = tb[:3, 3] + s * (tb[:3, 3] - ta[:3, 3])
        return out
    if len(prev) == 1:
        return known[prev[-1]].copy()
    return known[min(known)].copy()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def track(render, gt, p0, medium, t, unit, lr, marks, dev):
    dr = torch.zeros(3, device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dr, dt], lr=lr)

    def pose():
        d = torch.eye(4, device=dev)
        d[:3, :3] = E.axis_angle_to_matrix(dr)
        d[:3, 3] = dt / float(unit)
        return p0 @ d

    want, poses, secs = set(marks), {}, {}
    sync(dev)
    t0 = time.time()
    for i in range(1, max(marks) + 1):
        opt.zero_grad(set_to_none=True)
        F.l1_loss(render(pose(), medium, t=t), gt).backward()
        opt.step()
        if i in want:
            sync(dev)
            secs[i] = time.time() - t0
            poses[i] = pose().detach()
    return poses, secs


def device_lines(dev):
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    board = ""
    try:
        board = open("/proc/device-tree/model").read().strip("\x00\n ")
    except Exception:
        pass
    mode = ""
    if shutil.which("nvpmodel"):
        try:
            q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"NV Power Mode: *(\S+)", q)
            mode = m.group(1) if m else ""
        except Exception:
            pass
    return name, board, mode


def run(ctx):
    dev, tests, marks = ctx["dev"], ctx["tests"], sorted(set(int(k) for k in ctx["steps"] if int(k) > 0))
    nums = [E.fnum(x) for x in tests]
    order = np.argsort(nums)
    map_pose = {int(n): p for n, p in zip(ctx["map_nums"], ctx["map_poses"])}
    whole_peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tegra = Tegra()
    res = {"steps": marks, "frames": len(tests), "init": {}}
    final = {}
    for init in ctx["inits"]:
        known = dict(map_pose)
        rows = []
        w0 = time.time()
        for j in order:
            name, num = tests[j], nums[j]
            gt = ctx["load"](name)
            start = ctx["interp"][j] if init == "interp" else predict_pose(known, num)
            p0 = torch.tensor(start, dtype=torch.float32, device=dev)
            med = ctx["medium"](num, p0)
            t = num / ctx["fps"]
            poses, secs = track(ctx["render"], gt, p0, med, t, ctx["unit"], ctx["lr"], marks, dev)
            with torch.no_grad():
                sc = {0: ctx["score"](ctx["render"](p0, med, t=t), gt)}
                for k in marks:
                    sc[k] = ctx["score"](ctx["render"](poses[k], ctx["medium"](num, poses[k]), t=t), gt)
            known[num] = poses[marks[-1]].cpu().numpy().astype(np.float64)
            rows.append({"frame": num, "seconds": {k: secs[k] for k in marks}, "scores": sc})
            if init == ctx["inits"][0]:
                final[j] = (poses[marks[-1]], med, t)
            print("[bench] %s init, frame %06d: %d steps in %.2f s, PSNR %.2f -> %.2f" % (
                init, num, marks[-1], secs[marks[-1]], sc[0][0], sc[marks[-1]][0]), flush=True)
        w1 = time.time()
        table = []
        for k in [0] + marks:
            s = np.asarray([r["scores"][k] for r in rows])
            row = {"steps": k, "psnr": float(s[:, 0].mean()), "ssim": float(s[:, 1].mean()), "lpips": float(s[:, 2].mean())}
            if k:
                sec = np.asarray([r["seconds"][k] for r in rows[min(ctx["warmup"], len(rows) - 1):]])
                row.update(fps=float(1.0 / sec.mean()), mean_ms=float(1e3 * sec.mean()), median_ms=float(1e3 * np.median(sec)),
                           p90_ms=float(1e3 * np.percentile(sec, 90)))
            table.append(row)
        power = power_summary(tegra.window(w0, w1))
        res["init"][init] = {"table": table, "power": power, "rows": rows}
    with torch.no_grad():
        j0 = order[0]
        for _ in range(3):
            ctx["render"](final[j0][0], final[j0][1], t=final[j0][2])
        sync(dev)
        t0, count = time.time(), 0
        for j in order:
            for _ in range(ctx["render_reps"]):
                ctx["render"](final[j][0], final[j][1], t=final[j][2])
                count += 1
        sync(dev)
        el = time.time() - t0
    res["render"] = {"fps": count / el, "ms": 1e3 * el / count, "renders": count}
    ram = [r for r in tegra.window(0, time.time()) if "ram_used_mb" in r]
    tegra.stop()
    track_peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    res["memory"] = {"gpu_peak_gib": max(whole_peak, track_peak) / 2 ** 30, "gpu_peak_tracking_gib": track_peak / 2 ** 30,
                     "gpu_reserved_gib": (torch.cuda.max_memory_reserved() if dev.type == "cuda" else 0) / 2 ** 30,
                     "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2 ** 30}
    if ram:
        res["memory"]["system_ram_peak_mb"] = max(r["ram_used_mb"] for r in ram)
        res["memory"]["system_ram_total_mb"] = ram[-1]["ram_total_mb"]
    res["device"] = dict(zip(("gpu", "board", "power_mode"), device_lines(dev)))
    return res


def report(res, info):
    w = 86
    L = ["=" * w, " Swimm3R localization benchmark", "-" * w]
    d = res["device"]
    L.append(" device      %s%s%s | torch %s%s" % (
        d["gpu"], " | " + d["board"] if d["board"] else "", " | power mode " + d["power_mode"] if d["power_mode"] else "",
        torch.__version__, " | CUDA " + str(torch.version.cuda) if torch.version.cuda else ""))
    L.append(" scene       %s | %d map frames | %d held-out frames | %d x %d" % (
        info["scene"], info["map_frames"], res["frames"], info["hw"][0], info["hw"][1]))
    L.append(" map         %.2f M primitives (%s kernel%s) | %.1f MB as float32 tensors | map.npz %.1f MB" % (
        info["primitives"] / 1e6, info["kernel"], ", static part and dynamic copy per slot" if info["dynamic"] else "",
        info["map_mb"], info["map_file_mb"]))
    if info.get("mapping_seconds"):
        L.append(" mapping     %.0f s for %d frames (%.2f s per frame), run once before localization" % (
            info["mapping_seconds"], info["map_frames"], info["mapping_seconds"] / info["map_frames"]))
    names = {"interp": "interpolated between the neighbouring map frames", "motion": "constant velocity from the two previous frames (causal)"}
    for init, r in res["init"].items():
        L += ["-" * w, " Tracking    pose refinement by render and compare, one frame at a time",
              "             start pose " + names[init],
              "   steps  tracking FPS  ms/frame mean   median      p90     PSNR    SSIM   LPIPS"]
        for row in r["table"]:
            if row["steps"]:
                L.append("   %5d  %12.2f  %13.1f  %7.1f  %7.1f   %6.2f  %6.3f  %6.3f" % (
                    row["steps"], row["fps"], row["mean_ms"], row["median_ms"], row["p90_ms"], row["psnr"], row["ssim"], row["lpips"]))
            else:
                L.append("   %5d  %12s  %13s  %7s  %7s   %6.2f  %6.3f  %6.3f" % (0, "-", "-", "-", "-", row["psnr"], row["ssim"], row["lpips"]))
        last = r["table"][-1]
        near = next(x for x in r["table"][1:] if x["psnr"] >= last["psnr"] - 0.1)
        L.append("   within 0.1 dB of the final PSNR from %d steps on: tracking FPS %.2f" % (near["steps"], near["fps"]))
        if r["power"]:
            p = r["power"]
            L.append("   power %.1f W (%s, tegrastats mean of %d samples) | energy %.2f J per frame at %d steps" % (
                p["total_w"], " + ".join(p["sum_of"]), p["samples"], p["total_w"] * last["mean_ms"] / 1e3, last["steps"]))
    L += ["-" * w, " Rendering   %.1f FPS (%.2f ms per frame, %d x %d, mean of %d renders of the held-out views)" % (
        res["render"]["fps"], res["render"]["ms"], info["hw"][0], info["hw"][1], res["render"]["renders"]), "-" * w]
    m = res["memory"]
    L.append(" Footprint   GPU memory peak %.2f GiB allocated (%.2f GiB during tracking, %.2f GiB reserved)" % (
        m["gpu_peak_gib"], m["gpu_peak_tracking_gib"], m["gpu_reserved_gib"]))
    L.append("             process memory peak %.2f GiB (resident set)" % m["rss_peak_gib"])
    if "system_ram_peak_mb" in m:
        L.append("             system memory peak %d MB of %d MB (tegrastats, CPU and GPU share it)" % (
            m["system_ram_peak_mb"], m["system_ram_total_mb"]))
    L.append("             weights %s (the MASt3R base weights are not counted)" % " + ".join(
        "%s %.1f MB" % (k, v) for k, v in info["weights_mb"].items()))
    first = next(iter(res["init"].values()))["table"][-1]
    L += ["-" * w, " Summary     Tracking FPS %.2f (%d steps) | Rendering FPS %.1f | GPU memory %.2f GiB" % (
        first["fps"], first["steps"], res["render"]["fps"], m["gpu_peak_gib"]),
        " Tracking FPS is 1 / mean time of the pose refinement of one held-out frame. Image loading and scoring are not timed.",
        " The first %d frames of every pass are left out of the timing as warm-up." % info["warmup"], "=" * w]
    return "\n".join(L)
