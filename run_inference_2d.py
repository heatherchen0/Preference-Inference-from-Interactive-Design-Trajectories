from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from poster2d.utility import smooth_contrast_interaction_default_params
from src.config import (
    EvalConfig,
    ExperimentConfig,
    MCMCConfig,
    PlotConfig,
    PrefixConfig,
    to_dict,
)
from src.io_utils import load_jsonl, save_json
from src.plots import plot_prefix_metrics_learning_curves_multi_plotly
from src.seed_aggregation import (
    aggregate_prefix_metrics_df,
    build_prefix_metrics_long_df,
    plot_aggregate_prefix_metrics_plotly,
)


DEFAULT_DATASET = (
    "outputs_2d/um4_gamma0.95_alpha5_rep3/data/"
    "steps_UserModel4Boltzmann2D_gamma0.95_alpha5_train.jsonl"
)
UTILITY_KIND = "smooth_contrast_interaction"
_GAMMA_RE = re.compile(r"(?:^|[_-])gamma(?P<g>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)
_ALPHA_RE = re.compile(r"(?:^|[_-])alpha(?P<a>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed=(?P<seed>\d+)_mcmc=(?P<mcmc>\d+)$")
VARIANT_SET_CHOICES = ("all", "m4-gp")


def _parse_seed_ids(text: str) -> list[int]:
    seeds: list[int] = []
    seen = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            vals = range(int(lo_s), int(hi_s) + 1)
        else:
            vals = (int(part),)
        for v in vals:
            if v < 0:
                raise ValueError(f"Seed ids must be non-negative, got {v}.")
            if v not in seen:
                seen.add(v)
                seeds.append(v)
    if not seeds:
        raise ValueError("--seed-ids must contain at least one seed.")
    return seeds


def _parse_float_values(text: str) -> list[float]:
    values: list[float] = []
    seen = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        value = float(part)
        key = f"{value:g}"
        if key not in seen:
            seen.add(key)
            values.append(value)
    if not values:
        raise ValueError("Expected at least one comma-separated float value.")
    return values


def _fmt_param(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):g}"


def _timestamped_run_root(base: str | Path = "outputs_2d/inference") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base) / f"run_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _gamma_from_filename(path: Path, default: float = 0.95) -> float:
    m = _GAMMA_RE.search(path.name)
    return float(m.group("g")) if m else float(default)


def _alpha_from_filename(path: Path, default: float = 5.0) -> float:
    m = _ALPHA_RE.search(path.name)
    return float(m.group("a")) if m else float(default)


def _model_code(model_name: str) -> str:
    name = str(model_name).lower()
    if name == "model4_endpoint_discounted_boltzmann":
        return "m4"
    if name in {"birl_baseline", "birl_baseline_discounted_boltzmann"}:
        return "birl"
    if name.startswith("pbo"):
        return "pbo"
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def _likelihood_code(likelihood: str) -> str:
    value = str(likelihood).lower()
    if value == "boltzmann_qstar":
        return "boltz"
    if value == "start_end_logistic":
        return "start_end_logistic"
    if value == "transition_logistic":
        return "transition_logistic"
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")


def _variant_label(*, model_name: str, likelihood: str, prior: str) -> str:
    return f"{_model_code(model_name)}-{_likelihood_code(likelihood)}-{prior}"


def _variant_label_from_spec(v: dict) -> str:
    if "variant_label" in v:
        return str(v["variant_label"])
    return _variant_label(
        model_name=str(v.get("model_name", "")),
        likelihood=str(v.get("likelihood", "")),
        prior=str(v.get("prior", "")),
    )


def _variant_subdir_from_spec(v: dict) -> str:
    return f"v={_variant_label_from_spec(v)}"


def _base_traj_ids(rows: list[dict]) -> list[int]:
    seen = []
    known = set()
    for row in rows:
        tid = int(row["traj_id"])
        if tid not in known:
            known.add(tid)
            seen.append(tid)
    return seen


def _trajectory_order(traj_ids: list[int], *, seed: int) -> list[int]:
    arr = np.asarray(traj_ids, dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(arr)
    return [int(x) for x in arr.tolist()]


def _write_smoke_subset(*, dataset_path: Path, run_root: Path, n_traj: int) -> Path:
    rows = load_jsonl(dataset_path)
    keep = set(_base_traj_ids(rows)[: int(n_traj)])
    subset = [r for r in rows if int(r["traj_id"]) in keep]
    if not subset:
        raise ValueError("Smoke subset is empty.")

    out = run_root / "_smoke_data" / dataset_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def _all_variants() -> list[dict]:
    return [
        {
            "model_name": "model4_endpoint_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
        {
            "model_name": "model4_endpoint_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "iid",
        },
        {
            "model_name": "birl_baseline_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
        {
            "model_name": "birl_baseline_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "iid",
        },
        {
            "model_name": "pbo_baseline_start_end_preference",
            "likelihood": "start_end_logistic",
            "prior": "gp_rbf",
        },
        {
            "model_name": "pbo_baseline_start_end_preference",
            "likelihood": "start_end_logistic",
            "prior": "iid",
        },
        {
            "model_name": "pbo_baseline_transition_preference",
            "likelihood": "transition_logistic",
            "prior": "gp_rbf",
        },
        {
            "model_name": "pbo_baseline_transition_preference",
            "likelihood": "transition_logistic",
            "prior": "iid",
        },
    ]


def _m4_gp_variants(gp_ls_sd_values: list[float]) -> list[dict]:
    base = {
        "model_name": "model4_endpoint_discounted_boltzmann",
        "likelihood": "boltzmann_qstar",
        "prior": "gp_rbf",
    }
    base_label = _variant_label(**base)
    return [
        {
            **base,
            "gp_ls_sd": float(gp_ls_sd),
            "variant_label": f"{base_label}-ls{_fmt_param(gp_ls_sd)}",
        }
        for gp_ls_sd in gp_ls_sd_values
    ]


def _variants(*, variant_set: str, gp_ls_sd_values: list[float]) -> list[dict]:
    selected = str(variant_set).strip().lower()
    if selected == "all":
        return _all_variants()
    if selected == "m4-gp":
        return _m4_gp_variants(gp_ls_sd_values)
    raise ValueError(
        f"Unknown variant_set={variant_set!r}. Expected one of {VARIANT_SET_CHOICES}."
    )


def _base_config(*, grid_k: int, smoke: bool, seed: int) -> ExperimentConfig:
    utility_metadata = smooth_contrast_interaction_default_params(k=int(grid_k))
    mcmc = (
        MCMCConfig(draws=5, tune=5, chains=1, cores=1, target_accept=0.9, random_seed=seed)
        if smoke
        else MCMCConfig(draws=1000, tune=1000, chains=4, cores=4, target_accept=0.95, random_seed=seed)
    )
    prefix_mcmc = (
        MCMCConfig(draws=5, tune=5, chains=1, cores=1, target_accept=0.9, random_seed=seed)
        if smoke
        else MCMCConfig(draws=1000, tune=1000, chains=4, cores=4, target_accept=0.95, random_seed=seed)
    )
    prefix_list = (2, 4, 8) if smoke else (8, 16, 32, 64, 128, 256, 300)

    return ExperimentConfig(
        poster_min=1,
        poster_max=int(grid_k) * int(grid_k),
        state_space="2d",
        grid_k=int(grid_k),
        utility_kind=UTILITY_KIND,
        utility_metadata=utility_metadata,
        beta_fixed=20.0,
        show_plot=False,
        likelihood="boltzmann_qstar",
        prior="gp_rbf",
        gp_eta_sd=1.0,
        gp_ls_sd=0.25,
        birl_export_mode="stay",
        birl_reward_scale="discounted",
        birl_value_iters=50 if smoke else 100,
        mcmc_main=mcmc,
        prefix=PrefixConfig(
            enabled=True,
            use_main_mcmc=True,
            mcmc=prefix_mcmc,
            cache=True,
            overwrite_cache=True,
            train_schedule="explicit",
            train_k_traj_list=prefix_list,
        ),
        plots=PlotConfig(
            enabled=True,
            utility_curve=True,
            rank_plots=False,
            prefix_heatmap=True,
            prefix_slider=True,
        ),
        eval=EvalConfig(enabled=False),
    )


def _aggregate_run_root(run_root: Path) -> dict:
    rows = []
    prefix_entries = []

    for utility_dir in sorted(run_root.glob("utility=*")):
        if not utility_dir.is_dir():
            continue
        for dataset_dir in sorted(p for p in utility_dir.iterdir() if p.is_dir()):
            tables_dir = dataset_dir / "tables"
            plots_dir = dataset_dir / "plots"
            tables_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            dataset_entries = []
            for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                m = _SEED_RE.match(seed_dir.name)
                if not m:
                    continue
                seed = int(m.group("seed"))
                mcmc_seed = int(m.group("mcmc"))
                for variant_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("v=")):
                    summary_path = variant_dir / "summary.json"
                    prefix_path = variant_dir / "prefix_summaries.materialized.json"
                    if not summary_path.exists():
                        continue
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    label = variant_dir.name.split("=", 1)[1]
                    row = {
                        "utility_dir": utility_dir.name,
                        "dataset": dataset_dir.name,
                        "seed": seed,
                        "mcmc_seed": mcmc_seed,
                        "variant_label": label,
                        "variant_dir": str(variant_dir),
                        "summary_json": str(summary_path),
                    }
                    inputs = summary.get("inputs", {})
                    if "gamma_fixed" in inputs:
                        row["gamma_fixed"] = float(inputs["gamma_fixed"])
                    if "alpha_fixed" in inputs:
                        row["alpha_fixed"] = float(inputs["alpha_fixed"])
                    if "gp_eta_sd" in inputs:
                        row["gp_eta_sd"] = float(inputs["gp_eta_sd"])
                    if "gp_ls_sd" in inputs:
                        row["gp_ls_sd"] = float(inputs["gp_ls_sd"])
                    rec = summary.get("metrics", {}).get("recommendation", {})
                    curve = summary.get("metrics", {}).get("curve_error", {})
                    for key in ("simple_regret", "U_true_at_g_hat", "contrast_error", "contrast_profile_rmse"):
                        if key in rec:
                            row[key] = rec[key]
                        elif key in curve:
                            row[key] = curve[key]
                    rows.append(row)

                    if prefix_path.exists():
                        prefix_payload = json.loads(prefix_path.read_text(encoding="utf-8"))
                        dataset_entries.append(
                            {
                                "variant_label": label,
                                "seed": seed,
                                "mcmc_seed": mcmc_seed,
                                "trajectory_permutation_seed": seed,
                                "prefix_summaries": prefix_payload.get("prefix_summaries", []),
                            }
                        )

            if dataset_entries:
                prefix_entries.extend(dataset_entries)
                long_df = build_prefix_metrics_long_df(dataset_entries)
                if len(long_df):
                    long_path = tables_dir / "prefix_metrics_across_seeds_long.csv"
                    long_df.to_csv(long_path, index=False)
                    agg_df = aggregate_prefix_metrics_df(long_df)
                    agg_path = tables_dir / "prefix_metrics_across_seeds_mean.csv"
                    agg_df.to_csv(agg_path, index=False)
                    plot_aggregate_prefix_metrics_plotly(
                        agg_df,
                        out_path_html=plots_dir / "prefix_metrics_compare_seedmean_2d.html",
                        title="2D prefix metrics averaged across seeds",
                        metrics=(
                            "rmse",
                            "interval_coverage",
                            "gaussian_nlpd_raw",
                            "posterior_pairwise_acc_comp",
                            "simple_regret",
                            "contrast_error",
                            "contrast_profile_rmse",
                        ),
                    )

    if rows:
        import pandas as pd

        combined = pd.DataFrame(rows)
        out = run_root / "combined_run_summary_2d.csv"
        combined.to_csv(out, index=False)

    return {"n_rows": len(rows), "n_prefix_entries": len(prefix_entries)}


def run(args) -> Path:
    if (args.postprocess_only or args.aggregate_only) and args.run_root is None:
        raise ValueError("--run-root is required with --postprocess-only or --aggregate-only.")

    run_root = Path(args.run_root) if args.run_root else _timestamped_run_root()
    run_root.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        from src.postprocess import postprocess_run_root

        postprocess_run_root(run_root, utility_mode=UTILITY_KIND)
        _aggregate_run_root(run_root)
        return run_root

    if args.aggregate_only:
        _aggregate_run_root(run_root)
        return run_root

    dataset_path = Path(args.dataset)
    if args.smoke:
        dataset_path = _write_smoke_subset(
            dataset_path=dataset_path,
            run_root=run_root,
            n_traj=int(args.smoke_traj),
        )

    rows = load_jsonl(dataset_path)
    traj_ids = _base_traj_ids(rows)
    gamma_fixed = _gamma_from_filename(dataset_path)
    alpha_fixed = _alpha_from_filename(dataset_path)
    gp_ls_sd_values = _parse_float_values(args.gp_ls_sd_values)
    variants = _variants(
        variant_set=str(args.variant_set),
        gp_ls_sd_values=gp_ls_sd_values,
    )
    seed_ids = _parse_seed_ids(args.seed_ids)

    save_json(
        {
            "state_space": "2d",
            "grid_k": int(args.grid_k),
            "utility_kind": UTILITY_KIND,
            "dataset": str(dataset_path),
            "smoke": bool(args.smoke),
            "n_traj": len(traj_ids),
            "seed_ids": seed_ids,
            "base_mcmc_seed": int(args.base_mcmc_seed),
            "gamma_fixed": float(gamma_fixed),
            "alpha_fixed": float(alpha_fixed),
            "variant_set": str(args.variant_set),
            "gp_ls_sd_values": [float(v) for v in gp_ls_sd_values],
            "variants": variants,
        },
        run_root / "run_config.json",
    )

    dataset_run_dir = run_root / f"utility={UTILITY_KIND}" / dataset_path.stem
    dataset_run_dir.mkdir(parents=True, exist_ok=True)

    for seed in seed_ids:
        mcmc_seed = int(args.base_mcmc_seed) + int(seed)
        order = _trajectory_order(traj_ids, seed=int(seed))
        seed_run_dir = dataset_run_dir / f"seed={seed:03d}_mcmc={mcmc_seed}"
        seed_run_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "seed": int(seed),
                "mcmc_seed": int(mcmc_seed),
                "trajectory_order": order,
            },
            seed_run_dir / "trajectory_order.json",
        )

        cfg_seed_base = _base_config(grid_k=int(args.grid_k), smoke=bool(args.smoke), seed=mcmc_seed)
        prefix_by_label = {}
        from src.pipeline import run_experiment

        for variant in variants:
            label = _variant_label_from_spec(variant)
            variant_dir = seed_run_dir / _variant_subdir_from_spec(variant)
            cfg = replace(
                cfg_seed_base,
                model_name=variant["model_name"],
                likelihood=variant["likelihood"],
                prior=variant["prior"],
                alpha_fixed=float(alpha_fixed),
                gp_ls_sd=float(variant.get("gp_ls_sd", cfg_seed_base.gp_ls_sd)),
            )

            outputs = run_experiment(
                cfg=cfg,
                data_path=dataset_path,
                run_dir=variant_dir,
                gamma_fixed=float(gamma_fixed),
                hdi_prob=float(args.hdi_prob),
                trajectory_order=order,
                trajectory_permutation_seed=int(seed),
            )

            prefix_path = outputs.get("prefix_summaries_json")
            if prefix_path and Path(prefix_path).exists():
                payload = json.loads(Path(prefix_path).read_text(encoding="utf-8"))
                prefix_by_label[label] = payload.get("prefix_summaries", [])

        if prefix_by_label:
            plot_prefix_metrics_learning_curves_multi_plotly(
                prefix_by_label,
                out_path_html=seed_run_dir / "plots" / "prefix_metrics_compare_2d.html",
                title="2D prefix metrics by method",
                use_metrics="minmax",
            )

    _aggregate_run_root(run_root)
    return run_root


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2D poster inference experiments.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--grid-k", type=int, default=10)
    parser.add_argument("--seed-ids", default="0")
    parser.add_argument("--base-mcmc-seed", type=int, default=2000)
    parser.add_argument("--hdi-prob", type=float, default=0.94)
    parser.add_argument(
        "--variant-set",
        choices=VARIANT_SET_CHOICES,
        default="all",
        help="'all' preserves the existing 2D method set; 'm4-gp' sweeps Model 4 gp_rbf.",
    )
    parser.add_argument(
        "--gp-ls-sd-values",
        default="0.25",
        help="Comma-separated gp_ls_sd values used by --variant-set m4-gp.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-traj", type=int, default=8)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    root = run(parse_args())
    print(f"2D inference outputs written under: {root}")
