from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from run_inference import (
    _make_trajectory_order,
    _read_base_trajectory_ids,
    _variant_label_from_spec,
    _variant_subdir_from_spec,
    alpha_from_filename,
    gamma_from_filename,
)
from src.config import EvalConfig, ExperimentConfig, MCMCConfig, PlotConfig, PrefixConfig, to_dict
from src.io_utils import save_json
from src.pipeline import run_experiment


DATASETS = {
    "smooth": Path("steps_UserModel4Boltzmann_gamma0.95_alpha5_train_20260519_210144.jsonl"),
    "jagged": Path("steps_Jagged_UserModel4Boltzmann_gamma0.95_alpha5_train_20260519_211310.jsonl"),
}

VARIANTS = [
    {
        "model_name": "model4_endpoint_discounted_boltzmann_infer_alpha",
        "likelihood": "boltzmann_qstar",
        "prior": "gp_rbf",
        "variant_label": "m4-infer_alpha-boltz-gp_rbf",
    },
]


def _default_cores() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    return int(raw) if raw else 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one inferred-alpha 1D main task: one dataset, seed, variant."
    )
    parser.add_argument("--utility", choices=tuple(DATASETS), required=True)
    parser.add_argument("--seed-id", type=int, required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-mcmc-seed", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    return parser.parse_args()


def _base_config(args: argparse.Namespace, *, mcmc_seed: int) -> ExperimentConfig:
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
        birl_value_iters=100,
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
            train_schedule="geom2",
            train_geom_start=8,
            train_geom_ratio=2,
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
    )


def main() -> None:
    args = _parse_args()

    if args.list_variants:
        for idx, variant in enumerate(VARIANTS, start=1):
            print(idx, _variant_label_from_spec(variant), variant)
        return

    if int(args.seed_id) < 0:
        raise ValueError("--seed-id must be non-negative.")
    if not (1 <= int(args.variant_index) <= len(VARIANTS)):
        raise ValueError(f"--variant-index must be in 1..{len(VARIANTS)}.")

    utility_kind = str(args.utility).strip().lower()
    train_path = DATASETS[utility_kind]
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    seed = int(args.seed_id)
    mcmc_seed = int(args.base_mcmc_seed) + seed
    variant = dict(VARIANTS[int(args.variant_index) - 1])
    variant_label = _variant_label_from_spec(variant)

    base_traj_ids = _read_base_trajectory_ids(train_path)
    trajectory_order = _make_trajectory_order(base_traj_ids, seed=seed)

    dataset_run_dir = Path(args.run_root) / f"utility={utility_kind}" / train_path.stem
    seed_run_dir = dataset_run_dir / f"seed={seed:03d}_mcmc={mcmc_seed}"
    variant_run_dir = seed_run_dir / _variant_subdir_from_spec(variant)
    seed_run_dir.mkdir(parents=True, exist_ok=True)

    trajectory_order_path = seed_run_dir / "trajectory_order.json"
    save_json(
        {
            "train_file": str(train_path),
            "seed": seed,
            "trajectory_permutation_seed": seed,
            "mcmc_seed": mcmc_seed,
            "n_traj": int(len(trajectory_order)),
            "trajectory_order": trajectory_order,
        },
        trajectory_order_path,
    )

    cfg = replace(
        _base_config(args, mcmc_seed=mcmc_seed),
        utility_kind=utility_kind,
        model_name=str(variant["model_name"]),
        likelihood=str(variant["likelihood"]),
        prior=str(variant["prior"]),
        alpha_fixed=alpha_from_filename(train_path),
    )

    task_config = {
        "task_family": "infer_alpha_1d_main",
        "utility_kind": utility_kind,
        "train_file": str(train_path),
        "seed": seed,
        "mcmc_seed": mcmc_seed,
        "base_mcmc_seed": int(args.base_mcmc_seed),
        "trajectory_order_json": str(trajectory_order_path),
        "variant_index": int(args.variant_index),
        "variant_label": variant_label,
        "variant": variant,
        "run_root": str(args.run_root),
        "seed_run_dir": str(seed_run_dir),
        "variant_run_dir": str(variant_run_dir),
        "config": to_dict(cfg),
    }
    save_json(task_config, variant_run_dir / "triton_task_config.json")

    print("task started", flush=True)
    print(f"task_family=infer_alpha_1d_main", flush=True)
    print(f"utility_kind={utility_kind}", flush=True)
    print(f"train_file={train_path}", flush=True)
    print(f"seed={seed}", flush=True)
    print(f"mcmc_seed={mcmc_seed}", flush=True)
    print(f"variant_index={int(args.variant_index)}", flush=True)
    print(f"variant_label={variant_label}", flush=True)
    print(f"variant_run_dir={variant_run_dir}", flush=True)

    outputs = run_experiment(
        cfg=cfg,
        data_path=train_path,
        run_dir=variant_run_dir,
        gamma_fixed=gamma_from_filename(train_path),
        hdi_prob=0.94,
        trajectory_order=trajectory_order,
        trajectory_permutation_seed=seed,
    )

    save_json({**task_config, "outputs": outputs}, variant_run_dir / "triton_task_summary.json")

    print("task complete", flush=True)
    for key, value in sorted(outputs.items()):
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
