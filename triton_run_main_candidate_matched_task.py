from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import (
    EvalConfig,
    ExperimentConfig,
    MCMCConfig,
    PlotConfig,
    PosteriorArchiveConfig,
    PrefixConfig,
    to_dict,
)
from src.io_utils import save_json


DEFAULT_PREFIX_K_TRAJ_LIST = (8, 16, 32, 64, 128, 256, 500)
_GAMMA_RE = re.compile(r"(?:^|[_-])gamma(?P<g>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)
_ALPHA_RE = re.compile(r"(?:^|[_-])alpha(?P<a>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)


def _default_cores() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    return int(raw) if raw else 4


def _parse_prefix_list(text: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return tuple(vals) if vals else DEFAULT_PREFIX_K_TRAJ_LIST


def gamma_from_filename(path: Path, default: float = 0.9) -> float:
    m = _GAMMA_RE.search(Path(path).name)
    return float(m.group("g")) if m else float(default)


def alpha_from_filename(path: Path, default: float = 5.0) -> float:
    m = _ALPHA_RE.search(Path(path).name)
    return float(m.group("a")) if m else float(default)


def _jsonable_id(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _traj_sort_key(x):
    try:
        return (0, int(x))
    except Exception:
        return (1, str(x))


def _read_base_trajectory_ids(path: Path) -> list:
    seen = set()
    ids = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "traj_id" not in row:
                raise KeyError(f"Missing traj_id in row from {path}: {row}")
            tid = _jsonable_id(row["traj_id"])
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
    ids = sorted(ids, key=_traj_sort_key)
    if not ids:
        raise ValueError(f"No trajectories found in {path}")
    return ids


def _make_trajectory_order(base_traj_ids: list, *, seed: int) -> list:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(base_traj_ids))
    return [_jsonable_id(base_traj_ids[int(i)]) for i in perm]


def _model_code(model_name: str) -> str:
    name = str(model_name).lower()
    if name == "model4_endpoint_discounted_boltzmann":
        return "m4"
    if name == "birl_baseline_discounted_boltzmann":
        return "birl"
    if name in {
        "pbo_baseline",
        "pbo_baseline_start_end_preference",
        "pbo_baseline_transition_preference",
    }:
        return "pbo"
    return name


def _likelihood_code(likelihood: str) -> str:
    likelihood = str(likelihood).lower()
    if likelihood == "boltzmann_qstar":
        return "boltz"
    return likelihood


def _variant_label(*, prior: str, model_name: str, likelihood: str) -> str:
    return f"{_model_code(model_name)}-{_likelihood_code(likelihood)}-{prior}"


def _variant_subdir(*, prior: str, model_name: str, likelihood: str) -> Path:
    return Path(f"v={_variant_label(prior=prior, model_name=model_name, likelihood=likelihood)}")


def build_prefix_k_traj_list(
    n_traj: int,
    *,
    schedule: str,
    k_traj_list: list[int] | None = None,
) -> list[int]:
    n_traj = int(n_traj)
    if n_traj <= 0:
        raise ValueError(f"n_traj must be >= 1, got {n_traj}")
    schedule = str(schedule).strip().lower()
    if schedule not in {"explicit", "list"}:
        raise ValueError(f"This Triton runner expects explicit prefix schedule, got {schedule!r}.")
    ks = [] if k_traj_list is None else [int(k) for k in k_traj_list]
    ks = sorted({k for k in ks if 1 <= k <= n_traj})
    if not ks or ks[-1] != n_traj:
        ks.append(n_traj)
    return ks


def _read_tasks(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_task(tasks: list[dict], task_id: int) -> dict:
    matches = [row for row in tasks if int(row.get("task_id", -1)) == int(task_id)]
    if matches:
        if len(matches) > 1:
            raise ValueError(f"Duplicate task_id={task_id} in task CSV.")
        return matches[0]
    if 1 <= int(task_id) <= len(tasks):
        return tasks[int(task_id) - 1]
    raise ValueError(f"--task-id must be in 1..{len(tasks)}, got {task_id}.")


def _variant_label_from_task(row: dict) -> str:
    return _variant_label(
        prior=str(row["prior"]),
        model_name=str(row["model_name"]),
        likelihood=str(row["likelihood"]),
    )


def _require_float_match(name: str, observed: float, expected: float, *, tol: float = 1e-9) -> None:
    if abs(float(observed) - float(expected)) > tol:
        raise ValueError(
            f"{name} mismatch: task CSV has {expected:g}, filename encodes {observed:g}."
        )


def _validate_task(row: dict) -> dict:
    required = {
        "task_id",
        "dataset_label",
        "utility_kind",
        "train_file",
        "gamma_fixed",
        "alpha_fixed",
        "seed_id",
        "variant_index",
        "model_name",
        "likelihood",
        "prior",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise KeyError(f"Task row is missing required columns: {missing}")

    train_path = Path(str(row["train_file"]))
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    gamma_fixed = float(row["gamma_fixed"])
    alpha_fixed = float(row["alpha_fixed"])
    _require_float_match("gamma", gamma_from_filename(train_path), gamma_fixed)
    _require_float_match("alpha", alpha_from_filename(train_path), alpha_fixed)

    variant_label = _variant_label_from_task(row)
    csv_label = str(row.get("variant_label", "")).strip()
    if csv_label and csv_label != variant_label:
        raise ValueError(
            f"variant_label mismatch: task CSV has {csv_label!r}, spec implies {variant_label!r}."
        )

    seed = int(row["seed_id"])
    if seed < 0:
        raise ValueError(f"seed_id must be non-negative, got {seed}.")

    prefix_k = _parse_prefix_list(str(row.get("prefix_k_traj_list", "")))
    base_traj_ids = _read_base_trajectory_ids(train_path)
    resolved_prefix_k = build_prefix_k_traj_list(
        n_traj=len(base_traj_ids),
        schedule="explicit",
        k_traj_list=list(prefix_k),
    )
    if int(len(base_traj_ids)) != 500:
        raise ValueError(
            f"Expected 500 trajectories for first-batch main candidates, "
            f"got {len(base_traj_ids)} in {train_path}."
        )
    if resolved_prefix_k != list(DEFAULT_PREFIX_K_TRAJ_LIST):
        raise ValueError(
            "Resolved prefix schedule is not the expected "
            f"{list(DEFAULT_PREFIX_K_TRAJ_LIST)}: got {resolved_prefix_k}."
        )

    return {
        **row,
        "train_path": train_path,
        "gamma_fixed": gamma_fixed,
        "alpha_fixed": alpha_fixed,
        "seed": seed,
        "variant_label": variant_label,
        "prefix_k_traj_list": tuple(resolved_prefix_k),
        "base_traj_ids": base_traj_ids,
    }


def _base_config(args: argparse.Namespace, *, mcmc_seed: int, prefix_k_traj_list: tuple[int, ...]) -> ExperimentConfig:
    cores = int(args.cores) if args.cores is not None else _default_cores()

    return ExperimentConfig(
        poster_min=1,
        poster_max=100,
        beta_fixed=20.0,
        show_plot=False,
        likelihood="boltzmann_qstar",
        prior="gp_rbf",
        gp_eta_sd=1.0,
        gp_ls_sd=0.25,
        birl_export_mode="stay",
        birl_reward_scale="discounted",
        birl_value_iters=int(args.birl_value_iters),
        mcmc_main=MCMCConfig(
            draws=int(args.draws),
            tune=int(args.tune),
            chains=int(args.chains),
            cores=cores,
            target_accept=float(args.target_accept),
            random_seed=int(mcmc_seed),
        ),
        prefix=PrefixConfig(
            enabled=True,
            use_main_mcmc=True,
            cache=True,
            overwrite_cache=True,
            train_schedule="explicit",
            train_k_traj_list=tuple(int(k) for k in prefix_k_traj_list),
            train_traj_chunk_size=300,
            test_traj_chunk_size=30,
        ),
        plots=PlotConfig(
            enabled=not bool(args.disable_plots),
            utility_curve=True,
            rank_plots=False,
            prefix_heatmap=True,
            prefix_slider=True,
        ),
        eval=EvalConfig(enabled=False),
        posterior_archive=PosteriorArchiveConfig(
            save_u_draws=True,
            dtype="float32",
            save_diagnostics=True,
            save_full_idata=bool(args.save_full_idata),
        ),
    )


def _path_exists(value) -> bool:
    return bool(value) and Path(str(value)).exists()


def _save_json_atomic(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _check_expected_artifacts(outputs: dict, *, prefix_k_traj_list: tuple[int, ...]) -> dict:
    checks: dict[str, bool] = {
        "summary_json": _path_exists(outputs.get("summary_json")),
        "utility_per_state_csv": _path_exists(outputs.get("utility_per_state_csv")),
        "prefix_summaries_json": _path_exists(outputs.get("prefix_summaries_json")),
        "prefix_metrics_minmax_csv": _path_exists(outputs.get("prefix_metrics_minmax_csv")),
        "posterior_archive_manifest_json": _path_exists(
            outputs.get("posterior_archive_manifest_json")
        ),
    }

    archive = outputs.get("posterior_archive") or {}
    if isinstance(archive, dict):
        checks.update(
            {
                "archive_main_u_draws_npz": _path_exists(archive.get("main_u_draws_npz")),
                "archive_main_diagnostics_json": _path_exists(
                    archive.get("main_diagnostics_json")
                ),
                "archive_main_u_diagnostics_per_state_csv": _path_exists(
                    archive.get("main_u_diagnostics_per_state_csv")
                ),
                "archive_prefix_diagnostics_csv": _path_exists(
                    archive.get("prefix_diagnostics_csv")
                ),
            }
        )
    else:
        checks["posterior_archive_dict"] = False

    prefix_tables = outputs.get("prefix_per_state_tables_minmax_csvs") or []
    prefix_table_paths = [Path(str(p)) for p in prefix_tables]
    checks["prefix_table_count"] = len(prefix_table_paths) == len(prefix_k_traj_list)
    for k in prefix_k_traj_list:
        needle = f"prefix_{int(k):04d}_utility_per_state_minmax.csv"
        checks[f"prefix_table_{int(k):04d}"] = any(p.name == needle and p.exists() for p in prefix_table_paths)

    missing = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not missing,
        "checks": checks,
        "missing": missing,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one matched-alpha 1D main-candidate inference task from a task CSV."
    )
    parser.add_argument(
        "--task-csv",
        type=Path,
        default=Path("triton_main_candidate_matched_tasks.csv"),
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("results/main_1d_candidate_matched_seed0-4"))
    parser.add_argument("--base-mcmc-seed", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--birl-value-iters", type=int, default=100)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--save-full-idata", action="store_true")
    parser.add_argument("--skip-artifact-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = _read_tasks(Path(args.task_csv))

    if args.list_tasks:
        for row in tasks:
            print(
                row.get("task_id"),
                row.get("dataset_label"),
                f"seed={row.get('seed_id')}",
                row.get("variant_label"),
            )
        print(f"total_tasks={len(tasks)}")
        return

    if args.task_id is None:
        raise ValueError("--task-id is required unless --list-tasks is used.")

    task = _validate_task(_select_task(tasks, int(args.task_id)))
    train_path = Path(task["train_path"])
    seed = int(task["seed"])
    mcmc_seed = int(args.base_mcmc_seed) + seed
    prefix_k_traj_list = tuple(int(k) for k in task["prefix_k_traj_list"])
    utility_kind = str(task["utility_kind"]).strip().lower()

    trajectory_order = _make_trajectory_order(task["base_traj_ids"], seed=seed)

    dataset_run_dir = Path(args.run_root) / f"utility={utility_kind}" / train_path.stem
    seed_run_dir = dataset_run_dir / f"seed={seed:03d}_mcmc={mcmc_seed}"
    variant_run_dir = seed_run_dir / _variant_subdir(
        prior=str(task["prior"]),
        model_name=str(task["model_name"]),
        likelihood=str(task["likelihood"]),
    )

    cfg = replace(
        _base_config(args, mcmc_seed=mcmc_seed, prefix_k_traj_list=prefix_k_traj_list),
        utility_kind=utility_kind,
        model_name=str(task["model_name"]),
        likelihood=str(task["likelihood"]),
        prior=str(task["prior"]),
        alpha_fixed=float(task["alpha_fixed"]),
    )
    task_family = str(task.get("task_family") or "main_1d_candidate_matched")

    task_config = {
        "task_family": task_family,
        "task_csv": str(args.task_csv),
        "task_id": int(task["task_id"]),
        "dataset_index": int(task.get("dataset_index", 0) or 0),
        "dataset_label": str(task["dataset_label"]),
        "utility_kind": utility_kind,
        "train_file": str(train_path),
        "gamma_fixed": float(task["gamma_fixed"]),
        "alpha_fixed": float(task["alpha_fixed"]),
        "seed": seed,
        "trajectory_permutation_seed": seed,
        "mcmc_seed": mcmc_seed,
        "base_mcmc_seed": int(args.base_mcmc_seed),
        "n_traj": int(len(trajectory_order)),
        "prefix_k_traj_list": list(prefix_k_traj_list),
        "variant_index": int(task["variant_index"]),
        "variant_label": str(task["variant_label"]),
        "variant": {
            "model_name": str(task["model_name"]),
            "likelihood": str(task["likelihood"]),
            "prior": str(task["prior"]),
        },
        "run_root": str(args.run_root),
        "seed_run_dir": str(seed_run_dir),
        "variant_run_dir": str(variant_run_dir),
        "config": to_dict(cfg),
    }

    if bool(args.dry_run):
        print(json.dumps(task_config, indent=2))
        return

    from src.pipeline import run_experiment

    seed_run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_order_path = seed_run_dir / "trajectory_order.json"
    _save_json_atomic(
        {
            **{k: v for k, v in task_config.items() if k != "config"},
            "trajectory_order": trajectory_order,
        },
        trajectory_order_path,
    )
    task_config["trajectory_order_json"] = str(trajectory_order_path)
    save_json(task_config, variant_run_dir / "triton_task_config.json")

    print("task started", flush=True)
    print(f"task_family={task_family}", flush=True)
    print(f"task_id={int(task['task_id'])}", flush=True)
    print(f"dataset_label={task['dataset_label']}", flush=True)
    print(f"train_file={train_path}", flush=True)
    print(f"gamma_fixed={float(task['gamma_fixed']):g}", flush=True)
    print(f"alpha_fixed={float(task['alpha_fixed']):g}", flush=True)
    print(f"seed={seed}", flush=True)
    print(f"mcmc_seed={mcmc_seed}", flush=True)
    print(f"variant_index={int(task['variant_index'])}", flush=True)
    print(f"variant_label={task['variant_label']}", flush=True)
    print(f"prefix_k_traj_list={','.join(str(k) for k in prefix_k_traj_list)}", flush=True)
    print(f"variant_run_dir={variant_run_dir}", flush=True)

    started_at_utc = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    try:
        outputs = run_experiment(
            cfg=cfg,
            data_path=train_path,
            run_dir=variant_run_dir,
            gamma_fixed=float(task["gamma_fixed"]),
            hdi_prob=0.94,
            trajectory_order=trajectory_order,
            trajectory_permutation_seed=seed,
        )
    except Exception as exc:
        finished_at_utc = datetime.now(timezone.utc)
        runtime = {
            "started_at_utc": started_at_utc.isoformat(),
            "finished_at_utc": finished_at_utc.isoformat(),
            "elapsed_seconds": float(time.perf_counter() - started_perf),
        }
        save_json(
            {
                **task_config,
                "runtime": runtime,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            variant_run_dir / "triton_task_failure.json",
        )
        raise

    finished_at_utc = datetime.now(timezone.utc)
    runtime = {
        "started_at_utc": started_at_utc.isoformat(),
        "finished_at_utc": finished_at_utc.isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started_perf),
    }

    artifact_check = _check_expected_artifacts(
        outputs,
        prefix_k_traj_list=prefix_k_traj_list,
    )
    summary = {
        **task_config,
        "outputs": outputs,
        "runtime": runtime,
        "artifact_check": artifact_check,
    }
    save_json(summary, variant_run_dir / "triton_task_summary.json")

    if not bool(args.skip_artifact_check) and not bool(artifact_check["ok"]):
        raise RuntimeError(
            "Task completed, but expected artifacts are missing: "
            + ", ".join(str(x) for x in artifact_check["missing"])
        )

    print("task complete", flush=True)
    print(f"elapsed_seconds={runtime['elapsed_seconds']:.3f}", flush=True)
    for key, value in sorted(outputs.items()):
        print(f"{key}: {value}", flush=True)
    print(f"artifact_check_ok={artifact_check['ok']}", flush=True)


if __name__ == "__main__":
    main()
