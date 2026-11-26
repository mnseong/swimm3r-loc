import torch

LINALG = ("inv", "inv_ex", "solve", "solve_ex", "cholesky", "cholesky_ex", "svd", "svdvals", "eigh", "eigvalsh", "eig", "eigvals", "det",
          "slogdet", "pinv", "lstsq", "qr", "matrix_rank", "lu", "lu_factor", "lu_solve", "tensorinv", "tensorsolve", "matrix_power", "cond")
TOP = ("inverse", "det", "logdet", "slogdet", "pinverse", "svd", "cholesky", "cholesky_solve", "cholesky_inverse", "qr", "lstsq", "lu",
       "lu_solve", "matrix_power", "symeig")
METHODS = ("inverse", "det", "logdet", "slogdet", "pinverse", "svd", "cholesky", "cholesky_solve", "cholesky_inverse", "qr", "lu", "matrix_power")


def broken():
    if not torch.cuda.is_available():
        return False
    try:
        torch.linalg.inv(torch.eye(2, device="cuda"))
        return False
    except RuntimeError as e:
        return "linalg" in str(e) or "dlopen" in str(e) or "cusolver" in str(e)


def on_cpu(fn):
    def call(*a, **k):
        dev = next((x.device for x in list(a) + list(k.values()) if torch.is_tensor(x) and x.is_cuda), None)
        if dev is None:
            return fn(*a, **k)
        out = fn(*[x.cpu() if torch.is_tensor(x) else x for x in a], **{n: (v.cpu() if torch.is_tensor(v) else v) for n, v in k.items()})
        back = lambda o: o.to(dev) if torch.is_tensor(o) else o
        if isinstance(out, tuple):
            moved = [back(o) for o in out]
            try:
                return type(out)(moved)
            except Exception:
                return tuple(moved)
        return back(out)
    call.__name__ = getattr(fn, "__name__", "linalg")
    return call


def install(force=False):
    if not (force or broken()):
        return False
    for n in LINALG:
        if hasattr(torch.linalg, n):
            setattr(torch.linalg, n, on_cpu(getattr(torch.linalg, n)))
    for n in TOP:
        if hasattr(torch, n):
            setattr(torch, n, on_cpu(getattr(torch, n)))
    for n in METHODS:
        if hasattr(torch.Tensor, n):
            setattr(torch.Tensor, n, on_cpu(getattr(torch.Tensor, n)))
    print("[linalg] the CUDA linear algebra library of this torch build does not load on this device, "
          "matrix inverses and decompositions run on the CPU (small matrices only)", flush=True)
    return True
