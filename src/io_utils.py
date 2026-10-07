import json
import os
import re
from pathlib import Path
from datetime import datetime


def load_jsonl(file_path: Path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_idata(idata, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)


def make_run_root(base="results") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(base) / f"run_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def sanitize_label(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")


def extract_gamma_label(filename: str):
    m = re.search(r"gamma([0-9]*\.?[0-9]+)", filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
    
    
# import math

# def recommended_birl_value_iters(
#     gamma: float,
#     *,
#     reward_scale: str = "discounted",
#     tol: float = 1e-5,
#     min_iters: int = 25,
#     max_iters: int = 1000,
# ) -> int:
#     gamma = float(gamma)
#     if not (0.0 < gamma < 1.0):
#         raise ValueError(f"gamma must be in (0,1), got {gamma}")

#     reward_scale = str(reward_scale).strip().lower()

#     if reward_scale in ("discounted", "scaled", "one_minus_gamma", "1-gamma"):
#         v_bound = 1.0
#     elif reward_scale in ("raw", "unscaled", "none"):
#         v_bound = 1.0 / (1.0 - gamma)
#     else:
#         raise ValueError(f"Unknown reward_scale={reward_scale!r}")

#     k = math.ceil(math.log(float(tol) / v_bound) / math.log(gamma))

#     return int(max(min_iters, min(max_iters, k)))


# print(recommended_birl_value_iters(
#     gamma=0.95,
#     reward_scale="discounted",
#     tol=1e-5,
# ))
