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
)
from src.config import EvalConfig, ExperimentConfig, MCMCConfig, PlotConfig, PrefixConfig, to_dict
from src.io_utils import save_json
from src.pipeline import run_experiment


ROBUSTNESS_PREFIX_K_TRAJ_LIST = (5, 10, 20, 40, 60, 80, 100)

DATASETS = [
    {
        "dataset_label": "M2-gamma0.95-smooth",
        "train": Path("steps_UserModel2Discounted_gamma0.95_train.jsonl"),
        "gamma_fixed": 0.95,
        "alpha_fixed": 5.0,
        "gen_model_name": "model2_discounted",
        "generator_gamma": 0.95,
    },
    {
        "dataset_label": "M3-L5-smooth",
        "train": Path("steps_UserModel3Lookahead_L5_train.jsonl"),
        "gamma_fixed": 0.95,
        "alpha_fixed": 5.0,
        "gen_model_name": "model3_L_lookahead",
        "lookahead_L": 5,
    },
]

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
        description="Run one inferred-alpha 1D robustness task on Triton."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/robustness_1d"))
    parser.add_argument("--dataset-index", type=int, required=True)
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
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    return parser.parse_args()


def _base_config(args: argparse.Namespace, *, mcmc_seed: int) -> ExperimentConfig:
    cores = int(args.cores) if args.cores is not None else _default_cores()
    return ExperimentConfig(
        poster_min=1,
        poster_max=100,
        beta_fixed=20.0,
        show_plot=False,
        utility_kind="smooth",
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
            train_schedule="explicit",
            train_k_traj_list=ROBUSTNESS_PREFIX_K_TRAJ_LIST,
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

    if args.list_datasets:
        for idx, dataset in enumerate(DATASETS, start=1):
            print(idx, dataset["dataset_label"], Path(args.data_dir) / dataset["train"])
        return

    if args.list_variants:
        for idx, variant in enumerate(VARIANTS, start=1):
            print(idx, _variant_label_from_spec(variant), variant)
        return

    if not (1 <= int(args.dataset_index) <= len(DATASETS)):
        raise ValueError(f"--dataset-index must be in 1..{len(DATASETS)}.")
    if int(args.seed_id) < 0:
        raise ValueError("--seed-id must be non-negative.")
    if not (1 <= int(args.variant_index) <= len(VARIANTS)):
        raise ValueError(f"--variant-index must be in 1..{len(VARIANTS)}.")

    dataset = dict(DATASETS[int(args.dataset_index) - 1])
    variant = dict(VARIANTS[int(args.variant_index) - 1])
    train_path = Path(args.data_dir) / Path(dataset["train"])
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    seed = int(args.seed_id)
    mcmc_seed = int(args.base_mcmc_seed) + seed
    variant_label = _variant_label_from_spec(variant)

    base_traj_ids = _read_base_trajectory_ids(train_path)
    trajectory_order = _make_trajectory_order(base_traj_ids, seed=seed)

    dataset_run_dir = Path(args.run_root) / "utility=smooth" / train_path.stem
    seed_run_dir = dataset_run_dir / f"seed={seed:03d}_mcmc={mcmc_seed}"
    variant_run_dir = seed_run_dir / _variant_subdir_from_spec(variant)
    seed_run_dir.mkdir(parents=True, exist_ok=True)

    trajectory_order_path = seed_run_dir / "trajectory_order.json"
    save_json(
        {
            "dataset_suite": "robustness",
            "dataset_label": str(dataset["dataset_label"]),
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
        model_name=str(variant["model_name"]),
        likelihood=str(variant["likelihood"]),
        prior=str(variant["prior"]),
        alpha_fixed=float(dataset.get("alpha_fixed", 5.0)),
    )

    task_config = {
        "task_family": "infer_alpha_1d_robustness",
        "dataset_index": int(args.dataset_index),
        "dataset": {**dataset, "train": str(train_path)},
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
    print("task_family=infer_alpha_1d_robustness", flush=True)
    print(f"dataset_index={args.dataset_index}", flush=True)
    print(f"dataset_label={dataset['dataset_label']}", flush=True)
    print(f"train_file={train_path}", flush=True)
    print(f"seed={seed}", flush=True)
    print(f"mcmc_seed={mcmc_seed}", flush=True)
    print(f"variant_index={args.variant_index}", flush=True)
    print(f"variant_label={variant_label}", flush=True)
    print(f"variant_run_dir={variant_run_dir}", flush=True)

    outputs = run_experiment(
        cfg=cfg,
        data_path=train_path,
        run_dir=variant_run_dir,
        gamma_fixed=float(dataset.get("gamma_fixed", 0.95)),
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
