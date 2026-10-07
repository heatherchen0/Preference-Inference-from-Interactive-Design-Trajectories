from __future__ import annotations

from pathlib import Path
import json
import math
import csv
import hashlib
import numpy as np


ARCHIVE_SCHEMA = 3


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _as_jsonable(obj):
    return json.loads(json.dumps(obj, default=_json_default))


def _sha256_array(arr) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _posterior_u_array(idata, *, var_name: str = "U", dtype: str = "float32") -> np.ndarray:
    if not hasattr(idata, "posterior") or var_name not in idata.posterior:
        raise KeyError(f"Expected idata.posterior[{var_name!r}] to exist.")

    arr = np.asarray(idata.posterior[var_name].values)
    if arr.ndim < 3:
        raise ValueError(
            f"Expected posterior {var_name!r} dims (chain, draw, ...), got shape {arr.shape}."
        )

    chain, draw = arr.shape[:2]
    arr = np.asarray(arr, dtype=dtype)
    return arr.reshape((int(chain), int(draw), -1))


def _posterior_scalar_array(idata, *, var_name: str, dtype: str = "float32") -> np.ndarray:
    if not hasattr(idata, "posterior") or var_name not in idata.posterior:
        raise KeyError(f"Expected idata.posterior[{var_name!r}] to exist.")

    arr = np.asarray(idata.posterior[var_name].values)
    if arr.ndim < 2:
        raise ValueError(
            f"Expected posterior {var_name!r} dims (chain, draw), got shape {arr.shape}."
        )

    chain, draw = arr.shape[:2]
    tail = int(np.prod(arr.shape[2:]))
    if tail not in (0, 1):
        raise ValueError(
            f"Expected scalar posterior {var_name!r}, got shape {arr.shape}."
        )
    return np.asarray(arr, dtype=dtype).reshape((int(chain), int(draw)))


def save_u_draw_archive(
    *,
    idata,
    out_path: str | Path,
    posters,
    U_true,
    metadata: dict,
    var_name: str = "U",
    dtype: str = "float32",
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    draws = _posterior_u_array(idata, var_name=var_name, dtype=str(dtype))
    posters = np.asarray(posters, dtype=np.int64).reshape(-1)
    U_true = np.asarray(U_true, dtype=np.float64).reshape(-1)

    if draws.shape[2] != posters.shape[0]:
        raise ValueError(
            f"U draw state dimension {draws.shape[2]} does not match posters length {posters.shape[0]}."
        )
    if U_true.shape[0] != posters.shape[0]:
        raise ValueError(
            f"U_true length {U_true.shape[0]} does not match posters length {posters.shape[0]}."
        )

    U_sha256 = _sha256_array(draws)
    posters_sha256 = _sha256_array(posters)
    U_true_sha256 = _sha256_array(U_true)

    meta = {
        "schema": ARCHIVE_SCHEMA,
        "var_name": str(var_name),
        "dtype": str(np.dtype(draws.dtype)),
        "shape": [int(x) for x in draws.shape],
        "U_sha256": U_sha256,
        "posters_sha256": posters_sha256,
        "U_true_sha256": U_true_sha256,
        "archive_file_sha256": None,
        "archive_file_sha256_note": "Final file hash is stored in manifest entries after the file is written.",
        **_as_jsonable(metadata or {}),
    }
    metadata_json = json.dumps(meta, sort_keys=True)

    np.savez_compressed(
        out_path,
        U=draws,
        posters=posters,
        U_true=U_true,
        metadata_json=np.asarray(metadata_json),
    )

    archive_file_sha256 = _sha256_file(out_path)

    return {
        "path": str(out_path),
        "schema": ARCHIVE_SCHEMA,
        "var_name": str(var_name),
        "dtype": str(np.dtype(draws.dtype)),
        "shape": [int(x) for x in draws.shape],
        "n_chains": int(draws.shape[0]),
        "n_draws": int(draws.shape[1]),
        "n_states": int(draws.shape[2]),
        "U_sha256": U_sha256,
        "posters_sha256": posters_sha256,
        "U_true_sha256": U_true_sha256,
        "archive_file_sha256": archive_file_sha256,
    }


def load_u_draw_archive(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        U = np.asarray(data["U"])
        posters = np.asarray(data["posters"], dtype=int)
        U_true = np.asarray(data["U_true"], dtype=float)
        metadata_json = str(np.asarray(data["metadata_json"]).item())
    return {
        "path": str(path),
        "U": U,
        "posters": posters,
        "U_true": U_true,
        "metadata": json.loads(metadata_json),
        "archive_file_sha256": _sha256_file(path),
    }


def save_alpha_draw_archive(
    *,
    idata,
    out_path: str | Path,
    metadata: dict,
    dtype: str = "float32",
) -> dict:
    if not hasattr(idata, "posterior") or "alpha" not in idata.posterior:
        raise KeyError("Expected idata.posterior['alpha'] to exist.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = _posterior_scalar_array(idata, var_name="alpha", dtype=str(dtype))
    log_alpha = (
        _posterior_scalar_array(idata, var_name="log_alpha", dtype=str(dtype))
        if "log_alpha" in idata.posterior
        else np.log(np.asarray(alpha, dtype=str(dtype)))
    )
    if log_alpha.shape != alpha.shape:
        raise ValueError(
            f"log_alpha shape {log_alpha.shape} does not match alpha shape {alpha.shape}."
        )

    alpha_sha256 = _sha256_array(alpha)
    log_alpha_sha256 = _sha256_array(log_alpha)

    meta = {
        "schema": ARCHIVE_SCHEMA,
        "kind": "alpha_draws",
        "var_names": ["alpha", "log_alpha"],
        "dtype": str(np.dtype(alpha.dtype)),
        "shape": [int(x) for x in alpha.shape],
        "alpha_sha256": alpha_sha256,
        "log_alpha_sha256": log_alpha_sha256,
        "archive_file_sha256": None,
        "archive_file_sha256_note": "Final file hash is stored in manifest entries after the file is written.",
        **_as_jsonable(metadata or {}),
    }
    metadata_json = json.dumps(meta, sort_keys=True)

    np.savez_compressed(
        out_path,
        alpha=alpha,
        log_alpha=log_alpha,
        metadata_json=np.asarray(metadata_json),
    )

    archive_file_sha256 = _sha256_file(out_path)
    return {
        "path": str(out_path),
        "schema": ARCHIVE_SCHEMA,
        "kind": "alpha_draws",
        "var_names": ["alpha", "log_alpha"],
        "dtype": str(np.dtype(alpha.dtype)),
        "shape": [int(x) for x in alpha.shape],
        "n_chains": int(alpha.shape[0]),
        "n_draws": int(alpha.shape[1]),
        "alpha_sha256": alpha_sha256,
        "log_alpha_sha256": log_alpha_sha256,
        "archive_file_sha256": archive_file_sha256,
    }


def load_alpha_draw_archive(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        alpha = np.asarray(data["alpha"])
        log_alpha = np.asarray(data["log_alpha"])
        metadata_json = str(np.asarray(data["metadata_json"]).item())
    return {
        "path": str(path),
        "alpha": alpha,
        "log_alpha": log_alpha,
        "metadata": json.loads(metadata_json),
        "archive_file_sha256": _sha256_file(path),
    }


def _manifest_entry_with_checksum(entry: dict, *, archive_dir: Path) -> dict:
    out = dict(entry)
    path_value = out.get("path")
    if path_value and "archive_file_sha256" not in out:
        p = Path(str(path_value))
        if not p.exists():
            candidate = archive_dir / p
            if candidate.exists():
                p = candidate
        if p.exists() and p.is_file():
            out["archive_file_sha256"] = _sha256_file(p)
    return out


def write_archive_manifest(
    *,
    archive_dir: str | Path,
    entries: list[dict],
    metadata: dict | None = None,
) -> str:
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / "manifest.json"
    payload = {
        "schema": ARCHIVE_SCHEMA,
        "metadata": _as_jsonable(metadata or {}),
        "entries": [
            _as_jsonable(_manifest_entry_with_checksum(e, archive_dir=archive_dir))
            for e in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def read_archive_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_values(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nan_summary(arr, *, reducer, default=float("nan")) -> float:
    vals = _finite_values(arr)
    if vals.size == 0:
        return float(default)
    return float(reducer(vals))


def _worst_index(arr, *, mode: str) -> int | None:
    vals = np.asarray(arr, dtype=float).reshape(-1)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None
    masked = np.where(finite, vals, -np.inf if mode == "max" else np.inf)
    if mode == "max":
        return int(np.argmax(masked))
    if mode == "min":
        return int(np.argmin(masked))
    raise ValueError(f"Unknown mode={mode!r}")


def _sample_stats_array(idata, names: tuple[str, ...]):
    if not hasattr(idata, "sample_stats"):
        return None
    for name in names:
        if name in idata.sample_stats:
            return np.asarray(idata.sample_stats[name].values)
    return None


def summarize_idata_diagnostics(
    idata,
    *,
    var_name: str = "U",
    posters=None,
) -> tuple[dict, list[dict]]:
    import arviz as az

    if not hasattr(idata, "posterior") or var_name not in idata.posterior:
        raise KeyError(f"Expected idata.posterior[{var_name!r}] to exist.")

    u = np.asarray(idata.posterior[var_name].values)
    n_chains = int(u.shape[0])
    n_draws = int(u.shape[1])
    n_samples = int(n_chains * n_draws)
    n_states = int(np.prod(u.shape[2:]))
    posters_arr = (
        np.asarray(posters, dtype=int).reshape(-1)
        if posters is not None
        else np.arange(1, n_states + 1, dtype=int)
    )

    diag: dict[str, object] = {
        "var_name": str(var_name),
        "n_chains": n_chains,
        "n_draws": n_draws,
        "n_samples": n_samples,
        "n_states": n_states,
    }

    div = _sample_stats_array(idata, ("diverging", "divergence"))
    if div is not None:
        div = np.asarray(div, dtype=bool)
        diag["divergence_count"] = int(np.sum(div))
        diag["divergence_rate"] = float(np.mean(div))
    else:
        diag["divergence_count"] = None
        diag["divergence_rate"] = None

    accept = _sample_stats_array(idata, ("acceptance_rate", "accept_stat"))
    diag["mean_acceptance_rate"] = (
        _nan_summary(accept, reducer=np.mean) if accept is not None else None
    )

    tree_depth = _sample_stats_array(idata, ("tree_depth", "treedepth"))
    if tree_depth is not None:
        max_tree_depth = int(np.nanmax(tree_depth))
        diag["max_tree_depth"] = max_tree_depth
        reached = _sample_stats_array(idata, ("reached_max_treedepth",))
        if reached is not None:
            diag["max_tree_depth_count"] = int(np.sum(np.asarray(reached, dtype=bool)))
        else:
            diag["max_tree_depth_count"] = int(np.sum(np.asarray(tree_depth) == max_tree_depth))
    else:
        diag["max_tree_depth"] = None
        diag["max_tree_depth_count"] = None

    try:
        bfmi = np.asarray(az.bfmi(idata), dtype=float)
        diag["bfmi_min"] = _nan_summary(bfmi, reducer=np.min)
        diag["bfmi_mean"] = _nan_summary(bfmi, reducer=np.mean)
    except Exception:
        diag["bfmi_min"] = None
        diag["bfmi_mean"] = None

    per_state_rows: list[dict] = []
    try:
        rhat = np.asarray(az.rhat(idata, var_names=[var_name])[var_name].values, dtype=float).reshape(-1)
    except Exception:
        rhat = np.full(n_states, np.nan)
    try:
        ess_bulk = np.asarray(
            az.ess(idata, var_names=[var_name], method="bulk")[var_name].values,
            dtype=float,
        ).reshape(-1)
    except Exception:
        ess_bulk = np.full(n_states, np.nan)
    try:
        ess_tail = np.asarray(
            az.ess(idata, var_names=[var_name], method="tail")[var_name].values,
            dtype=float,
        ).reshape(-1)
    except Exception:
        ess_tail = np.full(n_states, np.nan)

    diag["rhat_max"] = _nan_summary(rhat, reducer=np.max)
    diag["rhat_p95"] = _nan_summary(rhat, reducer=lambda x: np.percentile(x, 95))
    diag["ess_bulk_min"] = _nan_summary(ess_bulk, reducer=np.min)
    diag["ess_tail_min"] = _nan_summary(ess_tail, reducer=np.min)

    idx_rhat = _worst_index(rhat, mode="max")
    idx_bulk = _worst_index(ess_bulk, mode="min")
    idx_tail = _worst_index(ess_tail, mode="min")

    def _poster_at(idx):
        if idx is None or idx >= posters_arr.size:
            return None
        return int(posters_arr[idx])

    diag["rhat_max_idx"] = idx_rhat
    diag["rhat_max_poster"] = _poster_at(idx_rhat)
    diag["ess_bulk_min_idx"] = idx_bulk
    diag["ess_bulk_min_poster"] = _poster_at(idx_bulk)
    diag["ess_tail_min_idx"] = idx_tail
    diag["ess_tail_min_poster"] = _poster_at(idx_tail)

    for i in range(n_states):
        per_state_rows.append(
            {
                "idx": int(i),
                "poster": _poster_at(i),
                "rhat": float(rhat[i]) if i < rhat.size and math.isfinite(float(rhat[i])) else float("nan"),
                "ess_bulk": float(ess_bulk[i]) if i < ess_bulk.size and math.isfinite(float(ess_bulk[i])) else float("nan"),
                "ess_tail": float(ess_tail[i]) if i < ess_tail.size and math.isfinite(float(ess_tail[i])) else float("nan"),
            }
        )

    return diag, per_state_rows


def write_json(obj, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_jsonable(obj), indent=2), encoding="utf-8")
    return str(path)


def write_rows_csv(rows: list[dict], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_as_jsonable(r) for r in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)
