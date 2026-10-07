from __future__ import annotations
from pathlib import Path
import argparse
import re
import pandas as pd
import json
import numpy as np
from datetime import datetime
from dataclasses import replace
from src.config import ExperimentConfig, MCMCConfig, PrefixConfig, PlotConfig, EvalConfig, to_dict
from src.io_utils import make_run_root, save_json, load_jsonl
from src.pipeline import run_experiment
from src.plots import (
    plot_prefix_metrics_learning_curves_multi_plotly,
    plot_posterior_U_individual_pngs,
    plot_posterior_U_overlay_plotly,
    plot_posterior_U_overlay_png,
    plot_posterior_U_minmax_overlay_plotly,
)
from src.seed_aggregation import (
    build_prefix_metrics_long_df,
    aggregate_prefix_metrics_df,
    aggregate_prefix_summaries_for_existing_plot,
    plot_aggregate_prefix_metrics_plotly,
    plot_aggregate_prefix_metric_pair_png,
    plot_aggregate_prefix_metric_pngs,
)
from src.postprocess import postprocess_run_root
from utility import get_utility


_GAMMA_RE = re.compile(r"(?:^|[_-])gamma(?P<g>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)
_ALPHA_RE = re.compile(r"(?:^|[_-])alpha(?P<a>[0-9]*\.?[0-9]+)(?:[_-]|\.|$)", re.IGNORECASE)
_SEED_DIR_RE = re.compile(r"^seed=(?P<seed>\d+)_mcmc=(?P<mcmc>\d+)$")
UTILITY_MODE_CHOICES = ("smooth", "jagged", "both")
DATASET_SUITE_CHOICES = ("main", "robustness")
VARIANT_SET_CHOICES = ("auto", "robustness", "robustness_alpha", "infer_alpha", "all")
ROBUSTNESS_PREFIX_K_TRAJ_LIST = (5, 10, 20, 40, 60, 80, 100)


def _parse_seed_ids(text: str) -> list[int]:
    seeds: list[int] = []
    seen = set()

    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue

        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid seed range {part!r}: upper bound is below lower bound.")
            vals = range(lo, hi + 1)
        else:
            vals = (int(part),)

        for v in vals:
            if v < 0:
                raise ValueError(f"Seed ids must be non-negative, got {v}.")
            if v not in seen:
                seen.add(v)
                seeds.append(v)

    if not seeds:
        raise ValueError("--seed-ids must contain at least one seed id.")

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


def _parse_int_values(text: str) -> list[int]:
    values: list[int] = []
    seen = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        value = int(part)
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("Expected at least one comma-separated integer value.")
    return values


def _fmt_param(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):g}"


def _make_aggregate_root(base: str | Path = "results") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"aggregate_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utility",
        choices=UTILITY_MODE_CHOICES,
        default="both",
        help=(
            "Which utility experiment to run. "
            "'smooth' runs only the smooth dataset, "
            "'jagged' runs only the jagged dataset, "
            "'both' runs smooth then jagged."
        ),
    )
    parser.add_argument(
        "--dataset-suite",
        choices=DATASET_SUITE_CHOICES,
        default="main",
        help=(
            "Dataset suite to run. 'main' uses the existing Model 4 smooth/jagged "
            "datasets; 'robustness' uses the Model 2 and Model 3 L-lookahead "
            "smooth robustness datasets."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help=(
            "Directory containing selected JSONL datasets. For robustness this "
            "usually points to data/robustness_1d."
        ),
    )
    parser.add_argument(
        "--variant-set",
        choices=VARIANT_SET_CHOICES,
        default="auto",
        help=(
            "'auto' uses the four robustness methods for --dataset-suite robustness "
            "and all configured variants otherwise. 'robustness' forces the four "
            "gp_rbf methods; 'robustness_alpha' runs Model 4 and BIRL across "
            "--alpha-values; 'infer_alpha' runs fixed-alpha and inferred-alpha "
            "Model 4 gp_rbf variants; 'all' runs all configured gp_rbf and iid variants."
        ),
    )
    parser.add_argument(
        "--robustness-model2-gammas",
        default="0.95",
        help=(
            "Comma-separated Model 2 robustness dataset gammas. "
            "Only used with --dataset-suite robustness."
        ),
    )
    parser.add_argument(
        "--robustness-model3-lookahead-Ls",
        default="5",
        help=(
            "Comma-separated Model 3 robustness lookahead horizons. "
            "Only used with --dataset-suite robustness."
        ),
    )
    parser.add_argument(
        "--alpha-values",
        default="5",
        help=(
            "Comma-separated alpha values for alpha-sweep variant sets such as "
            "robustness_alpha."
        ),
    )
    parser.add_argument(
        "--seed-ids",
        default="0",
        help=(
            "Trajectory/MCMC seed ids to run, using comma/range syntax. "
            "Examples: '0', '0,1,2', or '0-4'."
        ),
    )
    parser.add_argument(
        "--base-mcmc-seed",
        type=int,
        default=1000,
        help="MCMC seed is base_mcmc_seed + seed_id.",
    )
    parser.add_argument(
        "--mcmc-draws",
        type=int,
        default=None,
        help="Override main MCMC draws. Defaults to the thesis-run setting.",
    )
    parser.add_argument(
        "--mcmc-tune",
        type=int,
        default=None,
        help="Override main MCMC tuning draws. Defaults to the thesis-run setting.",
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=None,
        help="Override main MCMC chains. Defaults to the thesis-run setting.",
    )
    parser.add_argument(
        "--mcmc-cores",
        type=int,
        default=None,
        help="Override main MCMC cores. Defaults to the thesis-run setting.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Output run root. If omitted, a timestamped results/run_* directory "
            "is created. Reuse the same value across days to append more seeds."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not run inference. Scan completed seed directories and regenerate "
            "dataset-level across-seed tables/plots."
        ),
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Do not run inference. Recompute metrics/tables/plots from compact posterior "
            "archives under --run-root, then refresh aggregation."
        ),
    )
    parser.add_argument(
        "--aggregate-source-roots",
        nargs="+",
        default=None,
        help=(
            "Run roots to scan in --aggregate-only mode. Defaults to --run-root. "
            "Use this to combine seeds that were written to separate timestamped roots."
        ),
    )
    parser.add_argument(
        "--aggregate-output-root",
        type=Path,
        default=None,
        help=(
            "Where aggregate artifacts are written in --aggregate-only mode. "
            "Defaults to the single source root, or a new results/aggregate_* root "
            "when multiple source roots are provided."
        ),
    )
    return parser.parse_args()


def _resolve_dataset_path(
    data_dir: Path,
    stable_name: str,
    *,
    fallback_glob: str | None = None,
) -> Path:
    path = Path(data_dir) / stable_name
    if path.exists() or not fallback_glob:
        return path

    matches = sorted(
        Path(data_dir).glob(fallback_glob),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    return matches[0] if matches else path


def select_jobs(
    *,
    utility_mode: str,
    data_dir: Path,
    dataset_suite: str = "main",
    robustness_model2_gammas: list[float] | None = None,
    robustness_model3_lookahead_Ls: list[int] | None = None,
) -> list[dict]:
    data_dir = Path(data_dir)
    suite = str(dataset_suite).strip().lower()

    if suite == "main":
        all_jobs = [
            {
                "dataset_suite": "main",
                "utility_kind": "smooth",
                "train": data_dir / "steps_UserModel4Boltzmann_gamma0.95_alpha5_train_20260519_210144.jsonl",
            },
            {
                "dataset_suite": "main",
                "utility_kind": "jagged",
                "train": data_dir / "steps_Jagged_UserModel4Boltzmann_gamma0.95_alpha5_train_20260519_211310.jsonl",
            },
        ]
    elif suite == "robustness":
        model2_gammas = list(robustness_model2_gammas or [0.95])
        model3_Ls = list(robustness_model3_lookahead_Ls or [5])
        all_jobs = []

        for gamma in model2_gammas:
            gamma_label = _fmt_param(gamma)
            all_jobs.append(
                {
                    "dataset_suite": "robustness",
                    "dataset_label": f"M2-gamma{gamma_label}-smooth",
                    "utility_kind": "smooth",
                    "train": _resolve_dataset_path(
                        data_dir,
                        f"steps_UserModel2Discounted_gamma{gamma_label}_train.jsonl",
                        fallback_glob=f"steps_UserModel2Discounted_gamma{gamma_label}_train*.jsonl",
                    ),
                    "gamma_fixed": 0.95,
                    "alpha_fixed": 5.0,
                    "gen_model_name": "model2_discounted",
                    "generator_gamma": float(gamma),
                }
            )

        for lookahead_L in model3_Ls:
            all_jobs.append(
                {
                    "dataset_suite": "robustness",
                    "dataset_label": f"M3-L{int(lookahead_L)}-smooth",
                    "utility_kind": "smooth",
                    "train": _resolve_dataset_path(
                        data_dir,
                        f"steps_UserModel3Lookahead_L{int(lookahead_L)}_train.jsonl",
                        fallback_glob=f"steps_UserModel3Lookahead_L{int(lookahead_L)}_train*.jsonl",
                    ),
                    "gamma_fixed": 0.95,
                    "alpha_fixed": 5.0,
                    "gen_model_name": "model3_L_lookahead",
                    "lookahead_L": int(lookahead_L),
                }
            )
    else:
        raise ValueError(
            f"Unknown dataset_suite={dataset_suite!r}. "
            f"Expected one of {DATASET_SUITE_CHOICES}."
        )

    mode = str(utility_mode).strip().lower()

    if mode == "both":
        jobs = all_jobs
    else:
        jobs = [j for j in all_jobs if j["utility_kind"] == mode]

    if not jobs:
        raise ValueError(f"No jobs selected for utility_mode={utility_mode!r}")

    missing = [str(j["train"]) for j in jobs if not Path(j["train"]).exists()]
    if missing:
        raise FileNotFoundError(
            "The following selected dataset files do not exist:\n"
            + "\n".join(missing)
            + (
                "\n\nFor robustness, run: "
                "python run_experiments.py --suite robustness "
                "--out-dir data/robustness_1d"
                if suite == "robustness"
                else "\n\nEdit select_jobs() in run_inference.py or regenerate the datasets."
            )
        )

    return jobs

def alpha_from_filename(path: Path, default: float = 5.0) -> float:
    m = _ALPHA_RE.search(path.name)
    return float(m.group("a")) if m else float(default)


def infer_model_name_from_filename(path: Path, default: str = "model2_discounted") -> str:
    name = path.name.lower()
    if "model4" in name:
        return "model4_endpoint_discounted_boltzmann"
    if "usermodel2discounted" in name or "model2" in name:
        return "model2_discounted"
    if "usermodel3lookahead" in name or "model3" in name:
        return "model3_L_lookahead"
    return str(default)


def _dataset_label_from_stem(stem: str, utility_kind: str) -> str:
    s = str(stem)
    lower = s.lower()
    if "usermodel2discounted" in lower or "model2" in lower:
        gamma = _GAMMA_RE.search(s)
        suffix = "smooth" if str(utility_kind).lower() == "smooth" else str(utility_kind)
        if gamma:
            return f"M2-gamma{gamma.group('g')}-{suffix}"
        return "M2-smooth" if str(utility_kind).lower() == "smooth" else "M2"
    if "usermodel3lookahead" in lower or "model3" in lower:
        lookahead = re.search(r"(?:^|[_-])L(?P<L>\d+)(?:[_-]|\.|$)", s, re.IGNORECASE)
        suffix = "smooth" if str(utility_kind).lower() == "smooth" else str(utility_kind)
        if lookahead:
            return f"M3-L{lookahead.group('L')}-{suffix}"
        return "M3-smooth" if str(utility_kind).lower() == "smooth" else "M3"
    return s


def _dataset_suite_from_stem(stem: str) -> str:
    lower = str(stem).lower()
    if "usermodel2discounted" in lower or "usermodel3lookahead" in lower:
        return "robustness"
    return "main"


def gamma_from_filename(path: Path, default: float = 0.9) -> float:
    m = _GAMMA_RE.search(path.name)
    return float(m.group("g")) if m else float(default)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    """
    Read unique trajectory ids from the JSONL file.

    The returned order is deterministic: sorted by int(traj_id) if possible,
    otherwise by string.
    """
    rows = load_jsonl(path)
    seen = set()
    ids = []

    for r in rows:
        if "traj_id" not in r:
            raise KeyError(f"Missing traj_id in row from {path}: {r}")

        tid = _jsonable_id(r["traj_id"])
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)

    ids = sorted(ids, key=_traj_sort_key)

    if len(ids) == 0:
        raise ValueError(f"No trajectories found in {path}")

    return ids


def _make_trajectory_order(base_traj_ids: list, *, seed: int) -> list:
    """
    Create one trajectory-level permutation.

    This shuffles whole trajectories, not rows.
    """
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(base_traj_ids))
    return [_jsonable_id(base_traj_ids[int(i)]) for i in perm]


def _model_code(model_name: str) -> str:
    m = str(model_name).lower()
    if m in ("birl_baseline", "birl_baseline_discounted_boltzmann"):
        return "birl"
    if m in (
        "pbo_baseline",
        "pbo_baseline_start_end_preference",
        "pbo_baseline_transition_preference",
    ):
        return "pbo"
    if m == "model4_endpoint_discounted_boltzmann":
        return "m4"
    if m == "model4_endpoint_discounted_boltzmann_infer_alpha":
        return "m4-infer_alpha"
    if m == "model2_discounted":
        return "m2"
    return m.replace("model_", "m_").replace("endpoint_discounted_boltzmann", "edb")


def _likelihood_code(likelihood: str) -> str:
    l = str(likelihood).lower()
    if l == "boltzmann_qstar":
        return "boltz"
    return l


def _variant_code(*, prior: str, model_name: str, likelihood: str) -> str:
    return f"{_model_code(model_name)}-{_likelihood_code(likelihood)}-{str(prior)}"


def _variant_subdir(*, prior: str, model_name: str, likelihood: str) -> Path:
    return Path(f"v={_variant_code(prior=prior, model_name=model_name, likelihood=likelihood)}")


def _variant_label(*, prior: str, model_name: str, likelihood: str) -> str:
    return _variant_code(prior=prior, model_name=model_name, likelihood=likelihood)


def _variant_label_from_spec(v: dict) -> str:
    if "variant_label" in v:
        return str(v["variant_label"])
    return _variant_label(
        prior=str(v.get("prior", "")),
        model_name=str(v.get("model_name", v.get("infer_model_name", ""))),
        likelihood=str(v.get("likelihood", "")),
    )


def _variant_subdir_from_spec(v: dict) -> Path:
    return Path(f"v={_variant_label_from_spec(v)}")


def _all_variants() -> list[dict]:
    return [
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
    ]


def _robustness_variants() -> list[dict]:
    return [
        {
            "model_name": "model4_endpoint_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
        {
            "model_name": "birl_baseline_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
        {
            "model_name": "pbo_baseline_start_end_preference",
            "likelihood": "start_end_logistic",
            "prior": "gp_rbf",
        },
        {
            "model_name": "pbo_baseline_transition_preference",
            "likelihood": "transition_logistic",
            "prior": "gp_rbf",
        },
    ]


def _robustness_alpha_variants(alpha_values: list[float]) -> list[dict]:
    out = []
    base = [
        {
            "model_name": "model4_endpoint_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
        {
            "model_name": "birl_baseline_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
        },
    ]
    for alpha in alpha_values:
        alpha_label = _fmt_param(alpha)
        for spec in base:
            label = _variant_label(**spec)
            out.append(
                {
                    **spec,
                    "alpha_fixed": float(alpha),
                    "variant_label": f"{label}-alpha{alpha_label}",
                }
            )
    return out


def _infer_alpha_variants() -> list[dict]:
    return [
        {
            "model_name": "model4_endpoint_discounted_boltzmann",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
            "alpha_fixed": 5.0,
            "variant_label": "m4-boltz-gp_rbf-alpha5",
        },
        {
            "model_name": "model4_endpoint_discounted_boltzmann_infer_alpha",
            "likelihood": "boltzmann_qstar",
            "prior": "gp_rbf",
            "variant_label": "m4-infer_alpha-boltz-gp_rbf",
        },
    ]


def _select_variants(
    *,
    variant_set: str,
    dataset_suite: str,
    alpha_values: list[float] | None = None,
) -> tuple[str, list[dict]]:
    selected = str(variant_set).strip().lower()
    suite = str(dataset_suite).strip().lower()
    if selected == "auto":
        selected = "robustness" if suite == "robustness" else "all"

    if selected == "robustness":
        return selected, _robustness_variants()
    if selected == "robustness_alpha":
        return selected, _robustness_alpha_variants(alpha_values or [5.0])
    if selected == "infer_alpha":
        return selected, _infer_alpha_variants()
    if selected == "all":
        return selected, _all_variants()

    raise ValueError(
        f"Unknown variant_set={variant_set!r}. Expected one of {VARIANT_SET_CHOICES}."
    )


def _jsonable_job(job: dict) -> dict:
    out = {}
    for key, value in job.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _aggregate_seed_posterior_mean_series(entries: list[dict]) -> list[dict]:
    """
    Aggregate raw posterior mean U(g) curves across seeds by method.

    The band is +/- 1 SD of seed-level posterior means. It summarizes
    seed/MCMC stability, not pooled posterior uncertainty.
    """
    if not entries:
        return []

    out = []
    labels = sorted({str(e["variant_label"]) for e in entries})
    for label in labels:
        label_entries = [e for e in entries if str(e["variant_label"]) == label]
        means = [np.asarray(e["mean"], dtype=float).reshape(-1) for e in label_entries]
        if not means:
            continue

        lengths = {int(m.shape[0]) for m in means}
        if len(lengths) != 1:
            raise ValueError(
                f"Cannot aggregate posterior means for {label}: inconsistent lengths {sorted(lengths)}"
            )

        arr = np.stack(means, axis=0)
        mean = np.mean(arr, axis=0)
        sd = np.std(arr, axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros_like(mean)
        sem = sd / np.sqrt(float(arr.shape[0])) if arr.shape[0] >= 2 else np.zeros_like(mean)

        out.append(
            {
                "label": label,
                "mean": mean,
                "low": mean - sd,
                "high": mean + sd,
                "sd_low": mean - sd,
                "sd_high": mean + sd,
                "sem_low": mean - sem,
                "sem_high": mean + sem,
                "n_seeds": int(arr.shape[0]),
                "band": "seed_mean_sd",
            }
        )

    return out


def _parse_seed_dir_metadata(path: Path) -> tuple[int, int]:
    m = _SEED_DIR_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"Expected seed directory name like seed=000_mcmc=1000, got {path}")
    return int(m.group("seed")), int(m.group("mcmc"))


def _poster_grid_from_summary(summary: dict) -> tuple[int, int]:
    truth = summary.get("truth_utility", {})
    cfg = summary.get("config", {})
    return (
        int(truth.get("poster_min", cfg.get("poster_min", 1))),
        int(truth.get("poster_max", cfg.get("poster_max", 100))),
    )


def _variant_metadata_from_summary(summary: dict) -> tuple[str, dict]:
    inputs = summary.get("inputs", {})
    cfg = summary.get("config", {})

    model_name = str(inputs.get("model_name", cfg.get("model_name", "")))
    likelihood = str(inputs.get("likelihood", cfg.get("likelihood", "")))
    prior = str(inputs.get("prior", cfg.get("prior", "")))

    if not model_name or not likelihood or not prior:
        raise KeyError(
            "summary.json is missing model metadata needed for aggregation "
            "(inputs/config model_name, likelihood, prior)."
        )

    variant_label = _variant_label(
        prior=prior,
        model_name=model_name,
        likelihood=likelihood,
    )

    meta = {
        "variant_label": variant_label,
        "infer_model_name": model_name,
        "likelihood": likelihood,
        "prior": prior,
    }
    if "alpha_fixed" in inputs:
        meta["alpha_fixed"] = float(inputs["alpha_fixed"])
    if "alpha_mode" in inputs:
        meta["alpha_mode"] = str(inputs["alpha_mode"])
    if "alpha_prior_median" in inputs:
        meta["alpha_prior_median"] = float(inputs["alpha_prior_median"])
    if "alpha_prior_log_sd" in inputs:
        meta["alpha_prior_log_sd"] = float(inputs["alpha_prior_log_sd"])
    if "gp_ls_sd" in inputs:
        meta["gp_ls_sd"] = float(inputs["gp_ls_sd"])
    if "gp_eta_sd" in inputs:
        meta["gp_eta_sd"] = float(inputs["gp_eta_sd"])
    return variant_label, meta


def _alpha_columns_from_summary(summary: dict) -> dict:
    inputs = summary.get("inputs", {}) if isinstance(summary.get("inputs", {}), dict) else {}
    alpha_summary = (
        summary.get("alpha_summary", {})
        if isinstance(summary.get("alpha_summary", {}), dict)
        else {}
    )
    return {
        "alpha_mode": str(inputs.get("alpha_mode", "")),
        "alpha_post_mean": float(alpha_summary.get("mean", np.nan)),
        "alpha_post_median": float(alpha_summary.get("median", np.nan)),
        "alpha_post_sd": float(alpha_summary.get("sd", np.nan)),
        "alpha_post_q05": float(alpha_summary.get("q05", np.nan)),
        "alpha_post_q95": float(alpha_summary.get("q95", np.nan)),
        "alpha_post_low": float(alpha_summary.get("low", np.nan)),
        "alpha_post_high": float(alpha_summary.get("high", np.nan)),
        "alpha_prior_median": float(
            inputs.get("alpha_prior_median", alpha_summary.get("prior_median", np.nan))
        ),
        "alpha_prior_log_sd": float(
            inputs.get("alpha_prior_log_sd", alpha_summary.get("prior_log_sd", np.nan))
        ),
    }


def _variant_metadata_for_dir(summary: dict, variant_dir: Path) -> tuple[str, dict]:
    variant_label, meta = _variant_metadata_from_summary(summary)
    dirname = Path(variant_dir).name
    if dirname.startswith("v="):
        variant_label = dirname.split("=", 1)[1]
        meta["variant_label"] = variant_label
    return variant_label, meta


def _optional_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError as e:
        print(f"Warning: ignoring malformed optional JSON {path}: {e}", flush=True)
        return None


def _collect_existing_seed_outputs(
    source_roots: list[Path],
    *,
    utility_mode: str = "both",
) -> list[dict]:
    """
    Discover completed seed/variant runs from one or more run roots.

    This is the key post-hoc path: it uses only artifacts already on disk, so
    inference can be split across days and aggregated later.
    """
    mode = str(utility_mode).strip().lower()
    if mode not in UTILITY_MODE_CHOICES:
        raise ValueError(f"utility_mode must be one of {UTILITY_MODE_CHOICES}, got {utility_mode!r}")

    groups: dict[tuple[str, str], dict] = {}
    seen_variant_seed: dict[tuple[str, str, int, str], Path] = {}

    for source_root in source_roots:
        source_root = Path(source_root)
        if not source_root.exists():
            raise FileNotFoundError(f"Aggregate source root does not exist: {source_root}")

        for utility_dir in sorted(source_root.glob("utility=*")):
            if not utility_dir.is_dir():
                continue

            utility_from_dir = utility_dir.name.split("=", 1)[1].strip().lower()
            if mode != "both" and utility_from_dir != mode:
                continue

            for dataset_dir in sorted(p for p in utility_dir.iterdir() if p.is_dir()):
                for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                    if not _SEED_DIR_RE.match(seed_dir.name):
                        continue

                    seed_from_dir, mcmc_from_dir = _parse_seed_dir_metadata(seed_dir)
                    trajectory_order_path = seed_dir / "trajectory_order.json"
                    trajectory_order = _optional_json(trajectory_order_path)

                    for variant_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                        if not variant_dir.name.startswith("v="):
                            continue

                        summary_path = variant_dir / "summary.json"
                        if not summary_path.exists():
                            continue

                        summary = _read_json(summary_path)
                        inputs = summary.get("inputs", {})
                        cfg = summary.get("config", {})
                        seed_meta = summary.get("seed_metadata", {})

                        utility_kind = str(
                            inputs.get("utility_kind", cfg.get("utility_kind", utility_from_dir))
                        ).strip().lower()
                        if mode != "both" and utility_kind != mode:
                            continue

                        train_file = str(inputs.get("data_path", ""))
                        stem = Path(train_file).stem if train_file else dataset_dir.name
                        dataset_label = _dataset_label_from_stem(stem, utility_kind)
                        dataset_suite = _dataset_suite_from_stem(stem)
                        group_key = (utility_kind, stem)

                        variant_label, variant_meta = _variant_metadata_for_dir(summary, variant_dir)
                        duplicate_key = (utility_kind, stem, int(seed_from_dir), variant_label)
                        if duplicate_key in seen_variant_seed:
                            prev = seen_variant_seed[duplicate_key]
                            raise ValueError(
                                "Duplicate seed/variant output while aggregating. "
                                "Pass only one source root for each completed seed, or remove the duplicate.\n"
                                f"duplicate key={duplicate_key}\n"
                                f"first: {prev}\n"
                                f"second: {summary_path}"
                            )
                        seen_variant_seed[duplicate_key] = summary_path

                        poster_min, poster_max = _poster_grid_from_summary(summary)
                        gamma_fixed = float(inputs.get("gamma_fixed", gamma_from_filename(Path(train_file))))
                        alpha_fixed = float(inputs.get("alpha_fixed", alpha_from_filename(Path(train_file))))
                        gen_model_name = infer_model_name_from_filename(Path(train_file))

                        mcmc_seed = int(seed_meta.get("mcmc_main_random_seed", mcmc_from_dir))
                        trajectory_permutation_seed = seed_meta.get("trajectory_permutation_seed", seed_from_dir)
                        trajectory_permutation_seed = (
                            int(trajectory_permutation_seed)
                            if trajectory_permutation_seed is not None
                            else int(seed_from_dir)
                        )

                        group = groups.setdefault(
                            group_key,
                            {
                                "utility_kind": utility_kind,
                                "dataset_suite": dataset_suite,
                                "dataset_label": dataset_label,
                                "stem": stem,
                                "train_file": train_file,
                                "gamma_fixed": gamma_fixed,
                                "alpha_fixed": alpha_fixed,
                                "poster_min": poster_min,
                                "poster_max": poster_max,
                                "rows": [],
                                "prefix_entries": [],
                                "posterior_entries": [],
                                "variants_by_label": {},
                                "seed_ids": set(),
                                "mcmc_seeds": set(),
                                "source_roots": set(),
                                "dataset_source_dirs": set(),
                            },
                        )

                        if int(group["poster_min"]) != int(poster_min) or int(group["poster_max"]) != int(poster_max):
                            raise ValueError(
                                f"Poster grid mismatch for {group_key}: "
                                f"existing=({group['poster_min']}, {group['poster_max']}), "
                                f"new=({poster_min}, {poster_max}) from {summary_path}"
                            )

                        group["variants_by_label"][variant_label] = variant_meta
                        group["seed_ids"].add(int(seed_from_dir))
                        group["mcmc_seeds"].add(int(mcmc_seed))
                        group["source_roots"].add(str(source_root))
                        group["dataset_source_dirs"].add(str(dataset_dir))

                        prefix_path = variant_dir / "prefix_summaries.materialized.json"
                        outputs = {
                            "run_dir": str(variant_dir),
                            "summary_json": str(summary_path),
                        }

                        if prefix_path.exists():
                            outputs["prefix_summaries_json"] = str(prefix_path)

                        prefix_metrics_path = variant_dir / "tables" / "prefix_metrics_minmax.csv"
                        if prefix_metrics_path.exists():
                            outputs["prefix_metrics_minmax_csv"] = str(prefix_metrics_path)

                        alpha_prefix_path = variant_dir / "tables" / "alpha_prefix_summary.csv"
                        if alpha_prefix_path.exists():
                            outputs["alpha_prefix_summary_csv"] = str(alpha_prefix_path)

                        per_state_path = variant_dir / "tables" / "utility_per_state_raw.csv"
                        if per_state_path.exists():
                            outputs["utility_per_state_csv"] = str(per_state_path)

                        row = {
                            "dataset_suite": dataset_suite,
                            "dataset_label": dataset_label,
                            "utility_kind": utility_kind,
                            "train_file": train_file,
                            "stem": stem,
                            "seed": int(seed_from_dir),
                            "trajectory_permutation_seed": int(trajectory_permutation_seed),
                            "mcmc_seed": int(mcmc_seed),
                            "trajectory_order_json": (
                                str(trajectory_order_path)
                                if trajectory_order_path.exists()
                                else ""
                            ),
                            "seed_run_dir": str(seed_dir),
                            "gamma_fixed": float(gamma_fixed),
                            "alpha_fixed": float(alpha_fixed),
                            "gen_model_name": gen_model_name,
                            "infer_model_name": variant_meta["infer_model_name"],
                            "likelihood": variant_meta["likelihood"],
                            "prior": variant_meta["prior"],
                            "variant_label": variant_label,
                            "source_run_root": str(source_root),
                            **_alpha_columns_from_summary(summary),
                            **outputs,
                        }
                        group["rows"].append(row)

                        if prefix_path.exists():
                            mat = _read_json(prefix_path)
                            group["prefix_entries"].append(
                                {
                                    "utility_kind": utility_kind,
                                    "variant_label": variant_label,
                                    "seed": int(seed_from_dir),
                                    "mcmc_seed": int(mcmc_seed),
                                    "trajectory_permutation_seed": int(trajectory_permutation_seed),
                                    "prefix_summaries": mat.get("prefix_summaries", []),
                                }
                            )

                        U_raw_summary = summary.get("U_summary", {})
                        if "mean" in U_raw_summary:
                            group["posterior_entries"].append(
                                {
                                    "utility_kind": utility_kind,
                                    "variant_label": variant_label,
                                    "seed": int(seed_from_dir),
                                    "mcmc_seed": int(mcmc_seed),
                                    "trajectory_permutation_seed": int(trajectory_permutation_seed),
                                    "summary_json": str(summary_path),
                                    "mean": np.asarray(U_raw_summary["mean"], dtype=float),
                                }
                            )

                        if trajectory_order is not None:
                            row["trajectory_order_first_20"] = trajectory_order.get("trajectory_order", [])[:20]

    out = []
    for group in groups.values():
        group["seed_ids"] = sorted(group["seed_ids"])
        group["mcmc_seeds"] = sorted(group["mcmc_seeds"])
        group["source_roots"] = sorted(group["source_roots"])
        group["dataset_source_dirs"] = sorted(group["dataset_source_dirs"])
        base_candidates = {
            int(r["mcmc_seed"]) - int(r["seed"])
            for r in group["rows"]
            if "mcmc_seed" in r and "seed" in r
        }
        group["base_mcmc_seed"] = int(next(iter(base_candidates))) if len(base_candidates) == 1 else None
        out.append(group)

    return sorted(out, key=lambda g: (str(g["utility_kind"]), str(g["stem"])))


def _write_dataset_seed_aggregation(
    *,
    dataset_run_dir: Path,
    dataset_suite: str,
    dataset_label: str,
    utility_kind: str,
    train_file: str,
    stem: str,
    gamma_fixed: float,
    alpha_fixed: float,
    poster_min: int,
    poster_max: int,
    seed_ids: list[int],
    base_mcmc_seed: int | None,
    variants: list[dict],
    dataset_rows: list[dict],
    dataset_prefix_entries: list[dict],
    dataset_posterior_entries: list[dict],
) -> dict:
    dataset_run_dir = Path(dataset_run_dir)
    dataset_run_dir.mkdir(parents=True, exist_ok=True)
    true_utility = get_utility(utility_kind)
    dataset_combined: dict[str, str] = {}
    title_label = f"{dataset_label}: {stem}" if dataset_label and dataset_label != stem else stem

    if dataset_prefix_entries:
        tables_dir = dataset_run_dir / "tables"
        plots_dir = dataset_run_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        long_df = build_prefix_metrics_long_df(
            dataset_prefix_entries,
            use_metrics="minmax",
        )

        if len(long_df):
            long_csv = tables_dir / "prefix_metrics_across_seeds_long.csv"
            long_df.to_csv(long_csv, index=False)
            dataset_combined["prefix_metrics_across_seeds_long_csv"] = str(long_csv)

            agg_df = aggregate_prefix_metrics_df(long_df)

            if len(agg_df):
                agg_csv = tables_dir / "prefix_metrics_across_seeds_mean.csv"
                agg_df.to_csv(agg_csv, index=False)
                dataset_combined["prefix_metrics_across_seeds_mean_csv"] = str(agg_csv)

                p = plots_dir / "prefix_metrics_compare_seedmean_minmax.html"
                prefix_plot_metrics = (
                    "rmse",
                    "interval_coverage",
                    "gaussian_nlpd_raw",
                    "posterior_pairwise_acc_comp",
                    "simple_regret",
                )
                plot_aggregate_prefix_metrics_plotly(
                    agg_df,
                    out_path_html=p,
                    title="",
                    metrics=prefix_plot_metrics,
                    band="sem",
                )
                dataset_combined["compare_prefix_metrics_seedmean_minmax_html"] = str(p)
                metric_png_dir = plots_dir / "thesis_prefix_metrics"
                metric_pngs = plot_aggregate_prefix_metric_pngs(
                    agg_df,
                    out_dir=metric_png_dir,
                    metrics=prefix_plot_metrics,
                    band="sem",
                )
                if metric_pngs:
                    dataset_combined["thesis_prefix_metric_png_dir"] = str(metric_png_dir)
                    for metric, metric_path in metric_pngs.items():
                        dataset_combined[f"thesis_prefix_metric_{metric}_png"] = str(metric_path)
                pair_png = plots_dir / "thesis_interval_coverage_gaussian_nlpd.png"
                if plot_aggregate_prefix_metric_pair_png(
                    agg_df,
                    out_path_png=pair_png,
                    left_metric="interval_coverage",
                    right_metric="gaussian_nlpd_raw",
                    band="sem",
                ):
                    dataset_combined["thesis_interval_coverage_gaussian_nlpd_png"] = str(pair_png)

                mean_prefix_by_label = aggregate_prefix_summaries_for_existing_plot(
                    agg_df,
                    use_metrics="minmax",
                )
                if mean_prefix_by_label:
                    p2 = plots_dir / "prefix_metrics_compare_seedmean_existing_plotter_minmax.html"
                    plot_prefix_metrics_learning_curves_multi_plotly(
                        mean_prefix_by_label,
                        out_path_html=p2,
                        title="",
                        use_metrics="minmax",
                        include_pairwise_acc=True,
                    )
                    dataset_combined["compare_prefix_metrics_seedmean_existing_plotter_minmax_html"] = str(p2)

    if dataset_posterior_entries:
        plots_dir = dataset_run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        posterior_seedmean_series = _aggregate_seed_posterior_mean_series(dataset_posterior_entries)
        if posterior_seedmean_series:
            p = plots_dir / "posterior_U_compare_seedmean_raw.html"
            g_grid = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
            truth_y = np.asarray(true_utility(g_grid), dtype=float)

            plot_posterior_U_overlay_plotly(
                posterior_seedmean_series,
                poster_min=int(poster_min),
                poster_max=int(poster_max),
                out_path_html=p,
                title="",
                scale_mode="raw",
                yaxis_title="Utility",
                show_hdi_band=True,
                band_label="+/- 1 SD across seed-level posterior means",
                truth_y=truth_y,
                truth_label="Ground truth",
                truth_color="black",
                truth_dash="solid",
                truth_width=4,
            )
            dataset_combined["compare_posterior_U_seedmean_raw_html"] = str(p)
            p_png = plots_dir / "thesis_posterior_U_compare_seedmean_raw.png"
            if plot_posterior_U_overlay_png(
                posterior_seedmean_series,
                poster_min=int(poster_min),
                poster_max=int(poster_max),
                out_path_png=p_png,
                scale_mode="raw",
                yaxis_title="Utility",
                band_alpha=0.20,
                band_source="sd",
                show_hdi_band=True,
                truth_y=truth_y,
                truth_label="Ground truth",
                truth_color="black",
                truth_dash="solid",
                truth_width=2.6,
                line_width=1.45,
                legend_fontsize=9.0,
                legend_ncol=3,
                figsize=(10.5, 4.8),
            ):
                dataset_combined["thesis_posterior_U_seedmean_raw_png"] = str(p_png)
            p_individual_dir = plots_dir / "thesis_posterior_U_by_method"
            individual_pngs = plot_posterior_U_individual_pngs(
                posterior_seedmean_series,
                poster_min=int(poster_min),
                poster_max=int(poster_max),
                out_dir=p_individual_dir,
                scale_mode="raw",
                yaxis_title="Utility",
                band_alpha=0.20,
                band_source="sd",
                show_hdi_band=True,
                truth_y=truth_y,
                truth_label="Ground truth",
                truth_color="black",
                truth_dash="solid",
                truth_width=2.6,
                line_width=1.9,
                legend_fontsize=9.0,
            )
            if individual_pngs:
                dataset_combined["thesis_posterior_U_by_method_png_dir"] = str(p_individual_dir)

    for r in dataset_rows:
        r.update(dataset_combined)
        r["aggregate_dataset_run_dir"] = str(dataset_run_dir)

    variant_rows = []
    for v in sorted(variants, key=lambda x: str(x.get("variant_label", ""))):
        row = {
            "variant_label": _variant_label_from_spec(v),
            "infer_model_name": str(v.get("infer_model_name", v.get("model_name", ""))),
            "likelihood": str(v.get("likelihood", "")),
            "prior": str(v.get("prior", "")),
        }
        if "alpha_fixed" in v:
            row["alpha_fixed"] = float(v["alpha_fixed"])
        if "gp_ls_sd" in v:
            row["gp_ls_sd"] = float(v["gp_ls_sd"])
        variant_rows.append(row)
    variant_alpha_values = sorted(
        {
            float(v["alpha_fixed"])
            for v in variant_rows
            if "alpha_fixed" in v
        }
    )

    dataset_suite_summary = {
        "dataset_suite": str(dataset_suite),
        "dataset_label": str(dataset_label),
        "utility_kind": utility_kind,
        "train_file": str(train_file),
        "stem": stem,
        "gamma_fixed": float(gamma_fixed),
        "alpha_fixed": (
            float(variant_alpha_values[0])
            if len(variant_alpha_values) == 1
            else (float(alpha_fixed) if not variant_alpha_values else None)
        ),
        "alpha_values": variant_alpha_values,
        "seed_ids": [int(s) for s in sorted(seed_ids)],
        "base_mcmc_seed": (
            int(base_mcmc_seed)
            if base_mcmc_seed is not None
            else None
        ),
        "combined_artifacts": dataset_combined,
        "variants": variant_rows,
    }

    (dataset_run_dir / "suite_summary.json").write_text(
        json.dumps(dataset_suite_summary, indent=2),
        encoding="utf-8",
    )

    return dataset_combined


def _truth_y_from_summary(summary: dict, *, utility_kind: str, poster_min: int, poster_max: int) -> np.ndarray:
    curves = summary.get("curves", {})
    if isinstance(curves, dict) and "U_true" in curves:
        truth_y = np.asarray(curves["U_true"], dtype=float).reshape(-1)
        if truth_y.shape[0] == int(poster_max) - int(poster_min) + 1:
            return truth_y

    g_grid = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
    return np.asarray(get_utility(utility_kind)(g_grid), dtype=float)


def _refresh_variant_posterior_utility_plots_from_summary(
    *,
    variant_dir: Path,
    summary: dict,
    utility_kind: str,
    variant_label: str,
) -> dict[str, str]:
    poster_min, poster_max = _poster_grid_from_summary(summary)
    truth_y = _truth_y_from_summary(
        summary,
        utility_kind=utility_kind,
        poster_min=poster_min,
        poster_max=poster_max,
    )
    plots_dir = Path(variant_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    raw_summary = summary.get("U_summary", {})
    if isinstance(raw_summary, dict) and {"mean", "low", "high"}.issubset(raw_summary):
        raw_path = plots_dir / "posterior_utility_curve_raw.html"
        plot_posterior_U_overlay_plotly(
            [
                {
                    "label": "Posterior mean U",
                    "mean": np.asarray(raw_summary["mean"], dtype=float),
                    "low": np.asarray(raw_summary["low"], dtype=float),
                    "high": np.asarray(raw_summary["high"], dtype=float),
                }
            ],
            poster_min=int(poster_min),
            poster_max=int(poster_max),
            out_path_html=raw_path,
            title=f"Posterior utility curve, raw U(g)<br>{variant_label}",
            scale_mode="raw",
            yaxis_title="Utility U(g)",
            show_hdi_band=True,
            hdi_prob=float(summary.get("inputs", {}).get("hdi_prob", 0.94)),
            truth_y=truth_y,
            truth_label=f"Ground truth U(g), {utility_kind}",
            truth_color="black",
            truth_dash="solid",
            truth_width=4,
        )
        outputs["U_band_raw_html"] = str(raw_path)

    norm_summary = summary.get("U_summary_norm_draws", raw_summary)
    if isinstance(norm_summary, dict) and {"mean", "low", "high"}.issubset(norm_summary):
        minmax_path = plots_dir / "posterior_utility_curve.html"
        plot_posterior_U_overlay_plotly(
            [
                {
                    "label": "Posterior mean U",
                    "mean": np.asarray(norm_summary["mean"], dtype=float),
                    "low": np.asarray(norm_summary["low"], dtype=float),
                    "high": np.asarray(norm_summary["high"], dtype=float),
                    "already_normalized": "U_summary_norm_draws" in summary,
                    "normalization_method": (
                        norm_summary.get("normalization", {}).get("method", "")
                        if isinstance(norm_summary, dict)
                        else ""
                    ),
                }
            ],
            poster_min=int(poster_min),
            poster_max=int(poster_max),
            out_path_html=minmax_path,
            title=f"Posterior utility curve, posterior-draw min-max normalized<br>{variant_label}",
            scale_mode="posterior_draw_minmax",
            yaxis_title="Min-max normalized utility",
            show_hdi_band=True,
            hdi_prob=float(summary.get("inputs", {}).get("hdi_prob", 0.94)),
            truth_y=truth_y,
            truth_label=f"Ground truth U(g), {utility_kind}",
            truth_color="black",
            truth_dash="solid",
            truth_width=4,
        )
        outputs["U_band_minmax_html"] = str(minmax_path)

    return outputs


def _refresh_existing_html_plots(
    *,
    source_roots: list[Path],
    utility_mode: str = "both",
) -> list[dict]:
    mode = str(utility_mode).strip().lower()
    refreshed: list[dict] = []

    for source_root in [Path(p) for p in source_roots]:
        if not source_root.exists():
            continue

        for utility_dir in sorted(source_root.glob("utility=*")):
            if not utility_dir.is_dir():
                continue
            utility_kind = utility_dir.name.split("=", 1)[1].strip().lower()
            if mode != "both" and utility_kind != mode:
                continue

            true_utility = get_utility(utility_kind)
            for dataset_dir in sorted(p for p in utility_dir.iterdir() if p.is_dir()):
                for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                    if not _SEED_DIR_RE.match(seed_dir.name):
                        continue

                    seed, mcmc_seed = _parse_seed_dir_metadata(seed_dir)
                    prefix_by_label: dict[str, list[dict]] = {}
                    posterior_series_minmax = []
                    posterior_series_raw = []
                    poster_min = None
                    poster_max = None

                    for variant_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("v=")):
                        summary_path = variant_dir / "summary.json"
                        if not summary_path.exists():
                            continue
                        summary = _read_json(summary_path)
                        variant_label, _variant_meta = _variant_metadata_for_dir(summary, variant_dir)
                        variant_outputs = _refresh_variant_posterior_utility_plots_from_summary(
                            variant_dir=variant_dir,
                            summary=summary,
                            utility_kind=utility_kind,
                            variant_label=variant_label,
                        )
                        if variant_outputs:
                            refreshed.append(
                                {
                                    "scope": "variant",
                                    "variant_dir": str(variant_dir),
                                    "variant_label": variant_label,
                                    **variant_outputs,
                                }
                            )

                        prefix_path = variant_dir / "prefix_summaries.materialized.json"
                        if prefix_path.exists():
                            mat = _read_json(prefix_path)
                            prefix_by_label[variant_label] = mat.get("prefix_summaries", [])

                        cur_min, cur_max = _poster_grid_from_summary(summary)
                        poster_min = cur_min if poster_min is None else min(int(poster_min), int(cur_min))
                        poster_max = cur_max if poster_max is None else max(int(poster_max), int(cur_max))

                        raw_summary = summary.get("U_summary", {})
                        if isinstance(raw_summary, dict) and {"mean", "low", "high"}.issubset(raw_summary):
                            posterior_series_raw.append(
                                {
                                    "label": variant_label,
                                    "mean": np.asarray(raw_summary["mean"], dtype=float),
                                    "low": np.asarray(raw_summary["low"], dtype=float),
                                    "high": np.asarray(raw_summary["high"], dtype=float),
                                }
                            )

                        norm_summary = summary.get("U_summary_norm_draws", raw_summary)
                        if isinstance(norm_summary, dict) and {"mean", "low", "high"}.issubset(norm_summary):
                            posterior_series_minmax.append(
                                {
                                    "label": variant_label,
                                    "mean": np.asarray(norm_summary["mean"], dtype=float),
                                    "low": np.asarray(norm_summary["low"], dtype=float),
                                    "high": np.asarray(norm_summary["high"], dtype=float),
                                    "already_normalized": "U_summary_norm_draws" in summary,
                                    "normalization_method": (
                                        norm_summary.get("normalization", {}).get("method", "")
                                        if isinstance(norm_summary, dict)
                                        else ""
                                    ),
                                }
                            )

                    seed_outputs: dict[str, str] = {}
                    plots_dir = seed_dir / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    if prefix_by_label:
                        p = plots_dir / "prefix_metrics_compare_minmax.html"
                        plot_prefix_metrics_learning_curves_multi_plotly(
                            prefix_by_label,
                            out_path_html=p,
                            title=f"Prefix learning curves, seed={seed}, mcmc_seed={mcmc_seed}",
                            use_metrics="minmax",
                            include_pairwise_acc=True,
                        )
                        seed_outputs["seed_compare_prefix_metrics_minmax_html"] = str(p)

                    if poster_min is not None and poster_max is not None and posterior_series_minmax:
                        g_grid = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
                        truth_y = np.asarray(true_utility(g_grid), dtype=float)

                        p = plots_dir / "posterior_U_compare_minmax.html"
                        plot_posterior_U_minmax_overlay_plotly(
                            posterior_series_minmax,
                            poster_min=int(poster_min),
                            poster_max=int(poster_max),
                            out_path_html=p,
                            title=(
                                f"Posterior U(g) min-max overlay, utility={utility_kind}, "
                                f"seed={seed}, mcmc_seed={mcmc_seed}"
                            ),
                            show_hdi_band=True,
                            hdi_prob=0.94,
                            truth_y=truth_y,
                            truth_label=f"Ground truth U(g), {utility_kind}",
                            truth_color="black",
                            truth_dash="solid",
                            truth_width=4,
                        )
                        seed_outputs["seed_compare_posterior_U_minmax_html"] = str(p)

                    if poster_min is not None and poster_max is not None and posterior_series_raw:
                        g_grid = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
                        truth_y = np.asarray(true_utility(g_grid), dtype=float)

                        p_raw = plots_dir / "posterior_U_compare_raw.html"
                        plot_posterior_U_overlay_plotly(
                            posterior_series_raw,
                            poster_min=int(poster_min),
                            poster_max=int(poster_max),
                            out_path_html=p_raw,
                            title=(
                                f"Posterior U(g) raw overlay, utility={utility_kind}, "
                                f"seed={seed}, mcmc_seed={mcmc_seed}"
                            ),
                            scale_mode="raw",
                            yaxis_title="Utility U(g)",
                            show_hdi_band=True,
                            hdi_prob=0.94,
                            truth_y=truth_y,
                            truth_label=f"Ground truth U(g), {utility_kind}",
                            truth_color="black",
                            truth_dash="solid",
                            truth_width=4,
                        )
                        seed_outputs["seed_compare_posterior_U_raw_html"] = str(p_raw)

                    if seed_outputs:
                        suite_path = seed_dir / "suite_summary.json"
                        if suite_path.exists():
                            suite = _read_json(suite_path)
                            suite.setdefault("combined_artifacts", {}).update(seed_outputs)
                            suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
                        refreshed.append(
                            {
                                "scope": "seed",
                                "seed_dir": str(seed_dir),
                                **seed_outputs,
                            }
                        )

    return refreshed


def _aggregate_existing_seed_outputs(
    *,
    source_roots: list[Path],
    output_root: Path,
    utility_mode: str = "both",
) -> list[dict]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups = _collect_existing_seed_outputs(
        [Path(p) for p in source_roots],
        utility_mode=utility_mode,
    )

    all_rows: list[dict] = []
    aggregate_manifest = {
        "source_roots": [str(Path(p)) for p in source_roots],
        "output_root": str(output_root),
        "utility_mode": str(utility_mode),
        "datasets": [],
    }

    for group in groups:
        dataset_run_dir = output_root / f"utility={group['utility_kind']}" / str(group["stem"])

        dataset_combined = _write_dataset_seed_aggregation(
            dataset_run_dir=dataset_run_dir,
            dataset_suite=str(group.get("dataset_suite", _dataset_suite_from_stem(str(group["stem"])))),
            dataset_label=str(group.get("dataset_label", group["stem"])),
            utility_kind=str(group["utility_kind"]),
            train_file=str(group["train_file"]),
            stem=str(group["stem"]),
            gamma_fixed=float(group["gamma_fixed"]),
            alpha_fixed=float(group["alpha_fixed"]),
            poster_min=int(group["poster_min"]),
            poster_max=int(group["poster_max"]),
            seed_ids=list(group["seed_ids"]),
            base_mcmc_seed=group["base_mcmc_seed"],
            variants=list(group["variants_by_label"].values()),
            dataset_rows=group["rows"],
            dataset_prefix_entries=group["prefix_entries"],
            dataset_posterior_entries=group["posterior_entries"],
        )

        dataset_manifest = {
            "dataset_suite": str(group.get("dataset_suite", _dataset_suite_from_stem(str(group["stem"])))),
            "dataset_label": str(group.get("dataset_label", group["stem"])),
            "utility_kind": str(group["utility_kind"]),
            "stem": str(group["stem"]),
            "train_file": str(group["train_file"]),
            "output_dataset_dir": str(dataset_run_dir),
            "source_dataset_dirs": list(group["dataset_source_dirs"]),
            "seed_ids": [int(s) for s in group["seed_ids"]],
            "mcmc_seeds": [int(s) for s in group["mcmc_seeds"]],
            "n_variant_runs": int(len(group["rows"])),
            "combined_artifacts": dataset_combined,
        }

        save_json(dataset_manifest, dataset_run_dir / "aggregate_manifest.json")
        aggregate_manifest["datasets"].append(dataset_manifest)
        all_rows.extend(group["rows"])

    save_json(aggregate_manifest, output_root / "aggregate_manifest.json")
    return all_rows


def _write_combined_run_summary(rows: list[dict], run_root: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    sort_cols = [
        "dataset_suite",
        "dataset_label",
        "utility_kind",
        "gamma_fixed",
        "alpha_fixed",
        "alpha_mode",
        "seed",
        "infer_model_name",
        "likelihood",
        "prior",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]

    if len(df) and sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    df.to_csv(Path(run_root) / "combined_run_summary.csv", index=False)
    return df


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    utility_mode = str(args.utility).strip().lower()
    dataset_suite = str(args.dataset_suite).strip().lower()
    alpha_values = _parse_float_values(args.alpha_values)
    robustness_model2_gammas = _parse_float_values(args.robustness_model2_gammas)
    robustness_model3_lookahead_Ls = _parse_int_values(args.robustness_model3_lookahead_Ls)
    variant_set, variants = _select_variants(
        variant_set=str(args.variant_set),
        dataset_suite=dataset_suite,
        alpha_values=alpha_values,
    )

    if args.postprocess_only:
        if args.run_root is None:
            raise ValueError("--postprocess-only requires --run-root.")
        run_root = Path(args.run_root)
        if not run_root.exists():
            raise FileNotFoundError(f"--run-root does not exist: {run_root}")

        print(f"Postprocessing compact posterior archives under: {run_root}")
        postprocess_results = postprocess_run_root(
            run_root,
            utility_mode=utility_mode,
            hdi_prob=0.94,
        )
        refresh_results = _refresh_existing_html_plots(
            source_roots=[run_root],
            utility_mode=utility_mode,
        )
        postprocess_path = run_root / "postprocess_manifest.json"
        save_json(
            {
                "run_root": str(run_root),
                "utility_mode": utility_mode,
                "results": postprocess_results,
                "plot_refresh_results": refresh_results,
            },
            postprocess_path,
        )
        print(f"Postprocess manifest: {postprocess_path}")

        rows = _aggregate_existing_seed_outputs(
            source_roots=[run_root],
            output_root=run_root,
            utility_mode=utility_mode,
        )
        df = _write_combined_run_summary(rows, run_root)
        print(df)
        print(f"\nPostprocessed and refreshed aggregate results under: {run_root}")
        return

    if args.aggregate_only:
        source_roots = (
            [Path(p) for p in args.aggregate_source_roots]
            if args.aggregate_source_roots
            else ([Path(args.run_root)] if args.run_root is not None else [])
        )
        if not source_roots:
            raise ValueError(
                "--aggregate-only needs --run-root or --aggregate-source-roots."
            )

        if args.aggregate_output_root is not None:
            output_root = Path(args.aggregate_output_root)
        elif len(source_roots) == 1:
            output_root = Path(source_roots[0])
        else:
            output_root = _make_aggregate_root(base="results")

        print("Aggregating completed seeds from:")
        for p in source_roots:
            print(f"  {p}")
        print(f"Writing aggregate artifacts under: {output_root}")

        rows = _aggregate_existing_seed_outputs(
            source_roots=source_roots,
            output_root=output_root,
            utility_mode=utility_mode,
        )
        df = _write_combined_run_summary(rows, output_root)
        print(df)
        print(f"\nAggregate results saved under: {output_root}")
        return

    jobs = select_jobs(
        utility_mode=utility_mode,
        data_dir=data_dir,
        dataset_suite=dataset_suite,
        robustness_model2_gammas=robustness_model2_gammas,
        robustness_model3_lookahead_Ls=robustness_model3_lookahead_Ls,
    )

    # ---------------------------------------------------------------------
    # Multi-seed settings
    # ---------------------------------------------------------------------
    seed_ids = _parse_seed_ids(args.seed_ids)
    base_mcmc_seed = int(args.base_mcmc_seed)

    if args.run_root is None:
        run_root = make_run_root(base="results")
    else:
        run_root = Path(args.run_root)
        run_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving all outputs under: {run_root}")

    mcmc_draws = 1000 if args.mcmc_draws is None else int(args.mcmc_draws)
    mcmc_tune = 1000 if args.mcmc_tune is None else int(args.mcmc_tune)
    mcmc_chains = 4 if args.mcmc_chains is None else int(args.mcmc_chains)
    mcmc_cores = 4 if args.mcmc_cores is None else int(args.mcmc_cores)

    prefix_cfg = (
        PrefixConfig(
            enabled=True,
            use_main_mcmc=True,
            cache=True,
            overwrite_cache=True,
            train_schedule="explicit",
            train_k_traj_list=ROBUSTNESS_PREFIX_K_TRAJ_LIST,
            train_traj_chunk_size=300,
            test_traj_chunk_size=30,
        )
        if dataset_suite == "robustness"
        else PrefixConfig(
            enabled=True,
            use_main_mcmc=True,
            cache=True,
            overwrite_cache=True,
            train_schedule="geom2",
            train_geom_start=8,
            train_geom_ratio=2,
            train_traj_chunk_size=300,
            test_traj_chunk_size=30,
        )
    )

    cfg = ExperimentConfig(
        poster_min=1,
        poster_max=100,
        beta_fixed=20.0,
        show_plot=False,
        likelihood="boltzmann_qstar",
        prior="rw",
        gp_eta_sd=1.0,
        gp_ls_sd=0.25,
        
        birl_export_mode="stay",          # "stay" or "terminal"
        birl_reward_scale="discounted",   # "discounted" or "raw"
        birl_value_iters=100,

        mcmc_main=MCMCConfig(
            draws=mcmc_draws,
            tune=mcmc_tune,
            chains=mcmc_chains,
            cores=mcmc_cores,
            target_accept=0.95,
            random_seed=42,
        ),
        prefix=prefix_cfg,
        plots=PlotConfig(
            enabled=True,
            utility_curve=True,
            rank_plots=False,
            prefix_heatmap=True,
            prefix_slider=True,
        ),
        eval=EvalConfig(enabled=True),
    )

    save_json(
        {
            "base_cfg": to_dict(cfg),
            "utility_mode": utility_mode,
            "dataset_suite": dataset_suite,
            "data_dir": str(data_dir),
            "variant_set": variant_set,
            "alpha_values": [float(v) for v in alpha_values],
            "robustness_model2_gammas": [float(v) for v in robustness_model2_gammas],
            "robustness_model3_lookahead_Ls": [int(v) for v in robustness_model3_lookahead_Ls],
            "jobs": [
                _jsonable_job(j)
                for j in jobs
            ],
            "variants": variants,
            "seed_ids": seed_ids,
            "base_mcmc_seed": int(base_mcmc_seed),
            "seed_policy": {
                "trajectory_permutation_seed": "seed",
                "mcmc_seed": "base_mcmc_seed + seed",
            },
            "directory_policy": {
                "seed_dir": "seed={seed:03d}_mcmc={mcmc_seed}",
                "variant_dir": "v={variant_label}",
                "boltzmann_qstar_likelihood_code": "boltz",
            },
        },
        run_root / "run_config.json",
    )

    rows = []

    for j in jobs:
        utility_kind = str(j["utility_kind"]).strip().lower()
        true_utility = get_utility(utility_kind)

        train_path = Path(j["train"])

        gamma_fixed = float(j.get("gamma_fixed", gamma_from_filename(train_path)))
        alpha_fixed = float(j.get("alpha_fixed", alpha_from_filename(train_path)))
        gen_model_name = str(j.get("gen_model_name", infer_model_name_from_filename(train_path)))
        dataset_label = str(j.get("dataset_label", train_path.stem))

        dataset_run_dir = run_root / f"utility={utility_kind}" / train_path.stem
        dataset_run_dir.mkdir(parents=True, exist_ok=True)

        base_traj_ids = _read_base_trajectory_ids(train_path)

        dataset_rows = []
        dataset_prefix_entries = []
        dataset_posterior_entries = []

        permutation_manifest = {
            "dataset_suite": dataset_suite,
            "dataset_label": dataset_label,
            "utility_kind": utility_kind,
            "train_file": str(train_path),
            "stem": train_path.stem,
            "job": _jsonable_job(j),
            "base_traj_ids_sorted": base_traj_ids,
            "n_traj": int(len(base_traj_ids)),
            "seed_policy": {
                "trajectory_permutation_seed": "seed",
                "mcmc_seed": "base_mcmc_seed + seed",
                "base_mcmc_seed": int(base_mcmc_seed),
            },
            "seeds": [],
        }

        for seed in seed_ids:
            trajectory_permutation_seed = int(seed)
            mcmc_seed = int(base_mcmc_seed + seed)
            trajectory_order = _make_trajectory_order(
                base_traj_ids,
                seed=trajectory_permutation_seed,
            )

            seed_run_dir = dataset_run_dir / f"seed={seed:03d}_mcmc={mcmc_seed}"
            seed_run_dir.mkdir(parents=True, exist_ok=True)

            trajectory_order_path = seed_run_dir / "trajectory_order.json"
            save_json(
                {
                    "dataset_suite": dataset_suite,
                    "dataset_label": dataset_label,
                    "train_file": str(train_path),
                    "seed": int(seed),
                    "trajectory_permutation_seed": int(trajectory_permutation_seed),
                    "mcmc_seed": int(mcmc_seed),
                    "n_traj": int(len(trajectory_order)),
                    "trajectory_order": trajectory_order,
                },
                trajectory_order_path,
            )

            permutation_manifest["seeds"].append(
                {
                    "seed": int(seed),
                    "trajectory_permutation_seed": int(trajectory_permutation_seed),
                    "mcmc_seed": int(mcmc_seed),
                    "trajectory_order_json": str(trajectory_order_path),
                    "trajectory_order_first_20": trajectory_order[:20],
                }
            )

            print(
                f"\nDataset={train_path.stem} | seed={seed} | "
                f"traj_perm_seed={trajectory_permutation_seed} | mcmc_seed={mcmc_seed}"
            )

            cfg_seed_base = replace(
                cfg,
                mcmc_main=replace(
                    cfg.mcmc_main,
                    random_seed=mcmc_seed,
                ),
                prefix=replace(
                    cfg.prefix,
                    mcmc=replace(
                        cfg.prefix.mcmc,
                        random_seed=mcmc_seed,
                    ),
                ),
            )

            seed_rows = []
            prefix_by_label = {}
            posterior_series_minmax = []
            posterior_series_raw = []

            for v in variants:
                infer_model_name = str(v.get("model_name"))
                likelihood = str(v.get("likelihood"))
                prior = str(v.get("prior"))
                variant_alpha_fixed = float(v.get("alpha_fixed", alpha_fixed))

                variant_label = _variant_label_from_spec(v)

                variant_run_dir = seed_run_dir / _variant_subdir_from_spec(v)

                cfg_k = replace(
                    cfg_seed_base,
                    utility_kind=utility_kind,
                    model_name=infer_model_name,
                    likelihood=likelihood,
                    prior=prior,
                    alpha_fixed=float(variant_alpha_fixed),
                    eval=EvalConfig(
                        enabled=False,
                        obs_var_name="obs",
                    ),
                )

                print(f"  Running variant={variant_label} -> {variant_run_dir}")

                outputs = run_experiment(
                    cfg=cfg_k,
                    data_path=train_path,
                    run_dir=variant_run_dir,
                    gamma_fixed=gamma_fixed,
                    hdi_prob=0.94,
                    trajectory_order=trajectory_order,
                    trajectory_permutation_seed=trajectory_permutation_seed,
                )

                summary = _read_json(Path(outputs["summary_json"]))

                U_raw_summary = summary["U_summary"]
                U_plot_summary = summary.get("U_summary_norm_draws", U_raw_summary)

                posterior_series_minmax.append(
                    {
                        "label": variant_label,
                        "mean": np.asarray(U_plot_summary["mean"], dtype=float),
                        "low": np.asarray(U_plot_summary["low"], dtype=float),
                        "high": np.asarray(U_plot_summary["high"], dtype=float),
                        "already_normalized": "U_summary_norm_draws" in summary,
                        "normalization_method": (
                            U_plot_summary.get("normalization", {}).get("method", "")
                            if isinstance(U_plot_summary, dict)
                            else ""
                        ),
                    }
                )

                posterior_series_raw.append(
                    {
                        "label": variant_label,
                        "mean": np.asarray(U_raw_summary["mean"], dtype=float),
                        "low": np.asarray(U_raw_summary["low"], dtype=float),
                        "high": np.asarray(U_raw_summary["high"], dtype=float),
                    }
                )

                dataset_posterior_entries.append(
                    {
                        "utility_kind": utility_kind,
                        "variant_label": variant_label,
                        "seed": int(seed),
                        "mcmc_seed": int(mcmc_seed),
                        "trajectory_permutation_seed": int(trajectory_permutation_seed),
                        "summary_json": str(outputs["summary_json"]),
                        "mean": np.asarray(U_raw_summary["mean"], dtype=float),
                    }
                )

                if outputs.get("prefix_summaries_json"):
                    mat = _read_json(Path(outputs["prefix_summaries_json"]))
                    prefix_summaries = mat.get("prefix_summaries", [])
                    prefix_by_label[variant_label] = prefix_summaries

                    dataset_prefix_entries.append(
                        {
                            "utility_kind": utility_kind,
                            "variant_label": variant_label,
                            "seed": int(seed),
                            "mcmc_seed": int(mcmc_seed),
                            "trajectory_permutation_seed": int(trajectory_permutation_seed),
                            "prefix_summaries": prefix_summaries,
                        }
                    )

                row = {
                    "dataset_suite": dataset_suite,
                    "dataset_label": dataset_label,
                    "utility_kind": utility_kind,
                    "train_file": str(train_path),
                    "stem": train_path.stem,
                    "seed": int(seed),
                    "trajectory_permutation_seed": int(trajectory_permutation_seed),
                    "mcmc_seed": int(mcmc_seed),
                    "trajectory_order_json": str(trajectory_order_path),
                    "seed_run_dir": str(seed_run_dir),
                    "gamma_fixed": float(gamma_fixed),
                    "alpha_fixed": float(variant_alpha_fixed),
                    "gen_model_name": gen_model_name,
                    "infer_model_name": infer_model_name,
                    "likelihood": likelihood,
                    "prior": prior,
                    "variant_label": variant_label,
                    **_alpha_columns_from_summary(summary),
                    **outputs,
                }
                if "lookahead_L" in j:
                    row["lookahead_L"] = int(j["lookahead_L"])
                if "generator_gamma" in j:
                    row["generator_gamma"] = float(j["generator_gamma"])

                rows.append(row)
                dataset_rows.append(row)
                seed_rows.append(row)

            # -------------------------------------------------------------
            # Per-seed combined comparison plots
            # -------------------------------------------------------------
            seed_combined = {}
            plots_dir = seed_run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            if prefix_by_label:
                p = plots_dir / "prefix_metrics_compare_minmax.html"
                plot_prefix_metrics_learning_curves_multi_plotly(
                    prefix_by_label,
                    out_path_html=p,
                    title=(
                        f"Prefix learning curves, {dataset_label}, "
                        f"seed={seed}, mcmc_seed={mcmc_seed}"
                    ),
                    use_metrics="minmax",
                    include_pairwise_acc=True,
                )
                seed_combined["seed_compare_prefix_metrics_minmax_html"] = str(p)

            if posterior_series_minmax:
                p = plots_dir / "posterior_U_compare_minmax.html"

                g_grid = np.arange(int(cfg.poster_min), int(cfg.poster_max) + 1, dtype=float)
                truth_y = np.asarray(true_utility(g_grid), dtype=float)

                plot_posterior_U_minmax_overlay_plotly(
                    posterior_series_minmax,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=p,
                    title=(
                        f"Posterior U(g) min-max overlay, utility={utility_kind}, "
                        f"dataset={dataset_label}, seed={seed}, mcmc_seed={mcmc_seed}"
                    ),
                    show_hdi_band=True,
                    truth_y=truth_y,
                    truth_label=f"Ground truth U(g), {utility_kind}",
                    truth_color="black",
                    truth_dash="solid",
                    truth_width=4,
                )
                seed_combined["seed_compare_posterior_U_minmax_html"] = str(p)

                p_raw = plots_dir / "posterior_U_compare_raw.html"
                plot_posterior_U_overlay_plotly(
                    posterior_series_raw,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=p_raw,
                    title=(
                        f"Posterior U(g) raw overlay, utility={utility_kind}, "
                        f"dataset={dataset_label}, seed={seed}, mcmc_seed={mcmc_seed}"
                    ),
                    scale_mode="raw",
                    yaxis_title="Utility U(g)",
                    show_hdi_band=True,
                    hdi_prob=0.94,
                    truth_y=truth_y,
                    truth_label=f"Ground truth U(g), {utility_kind}",
                    truth_color="black",
                    truth_dash="solid",
                    truth_width=4,
                )
                seed_combined["seed_compare_posterior_U_raw_html"] = str(p_raw)

            for r in seed_rows:
                r.update(seed_combined)

            seed_variant_rows = []
            for v in variants:
                vr = {
                    "variant_label": _variant_label_from_spec(v),
                    "infer_model_name": str(v.get("model_name", gen_model_name)),
                    "likelihood": str(v.get("likelihood", cfg.likelihood)),
                    "prior": str(v.get("prior", cfg.prior)),
                }
                if "alpha_fixed" in v:
                    vr["alpha_fixed"] = float(v["alpha_fixed"])
                seed_variant_rows.append(vr)
            seed_alpha_values = sorted(
                {
                    float(vr["alpha_fixed"])
                    for vr in seed_variant_rows
                    if "alpha_fixed" in vr
                }
            )

            seed_suite_summary = {
                "dataset_suite": dataset_suite,
                "dataset_label": dataset_label,
                "utility_kind": utility_kind,
                "train_file": str(train_path),
                "stem": train_path.stem,
                "job": _jsonable_job(j),
                "seed": int(seed),
                "trajectory_permutation_seed": int(trajectory_permutation_seed),
                "mcmc_seed": int(mcmc_seed),
                "trajectory_order_json": str(trajectory_order_path),
                "gamma_fixed": float(gamma_fixed),
                "alpha_fixed": (
                    float(seed_alpha_values[0])
                    if len(seed_alpha_values) == 1
                    else (float(alpha_fixed) if not seed_alpha_values else None)
                ),
                "alpha_values": seed_alpha_values,
                "combined_artifacts": seed_combined,
                "variants": seed_variant_rows,
            }

            (seed_run_dir / "suite_summary.json").write_text(
                json.dumps(seed_suite_summary, indent=2),
                encoding="utf-8",
            )

        save_json(permutation_manifest, dataset_run_dir / "permutation_manifest.json")

        # -------------------------------------------------------------
        # Dataset-level aggregation across seeds
        # -------------------------------------------------------------
        dataset_combined = {}

        if dataset_prefix_entries:
            tables_dir = dataset_run_dir / "tables"
            plots_dir = dataset_run_dir / "plots"
            tables_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            long_df = build_prefix_metrics_long_df(
                dataset_prefix_entries,
                use_metrics="minmax",
            )

            if len(long_df):
                long_csv = tables_dir / "prefix_metrics_across_seeds_long.csv"
                long_df.to_csv(long_csv, index=False)
                dataset_combined["prefix_metrics_across_seeds_long_csv"] = str(long_csv)

                agg_df = aggregate_prefix_metrics_df(long_df)

                if len(agg_df):
                    agg_csv = tables_dir / "prefix_metrics_across_seeds_mean.csv"
                    agg_df.to_csv(agg_csv, index=False)
                    dataset_combined["prefix_metrics_across_seeds_mean_csv"] = str(agg_csv)

                    p = plots_dir / "prefix_metrics_compare_seedmean_minmax.html"
                    prefix_plot_metrics = (
                        "rmse",
                        "interval_coverage",
                        "gaussian_nlpd_raw",
                        "posterior_pairwise_acc_comp",
                        "simple_regret",
                    )
                    plot_aggregate_prefix_metrics_plotly(
                        agg_df,
                        out_path_html=p,
                        title="",
                        metrics=prefix_plot_metrics,
                        band="sem",
                    )
                    dataset_combined["compare_prefix_metrics_seedmean_minmax_html"] = str(p)
                    metric_png_dir = plots_dir / "thesis_prefix_metrics"
                    metric_pngs = plot_aggregate_prefix_metric_pngs(
                        agg_df,
                        out_dir=metric_png_dir,
                        metrics=prefix_plot_metrics,
                        band="sem",
                    )
                    if metric_pngs:
                        dataset_combined["thesis_prefix_metric_png_dir"] = str(metric_png_dir)
                        for metric, metric_path in metric_pngs.items():
                            dataset_combined[f"thesis_prefix_metric_{metric}_png"] = str(metric_path)
                    pair_png = plots_dir / "thesis_interval_coverage_gaussian_nlpd.png"
                    if plot_aggregate_prefix_metric_pair_png(
                        agg_df,
                        out_path_png=pair_png,
                        left_metric="interval_coverage",
                        right_metric="gaussian_nlpd_raw",
                        band="sem",
                    ):
                        dataset_combined["thesis_interval_coverage_gaussian_nlpd_png"] = str(pair_png)

                    mean_prefix_by_label = aggregate_prefix_summaries_for_existing_plot(
                        agg_df,
                        use_metrics="minmax",
                    )
                    if mean_prefix_by_label:
                        p2 = plots_dir / "prefix_metrics_compare_seedmean_existing_plotter_minmax.html"
                        plot_prefix_metrics_learning_curves_multi_plotly(
                            mean_prefix_by_label,
                            out_path_html=p2,
                            title="",
                            use_metrics="minmax",
                            include_pairwise_acc=True,
                        )
                        dataset_combined["compare_prefix_metrics_seedmean_existing_plotter_minmax_html"] = str(p2)

        if dataset_posterior_entries:
            plots_dir = dataset_run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            posterior_seedmean_series = _aggregate_seed_posterior_mean_series(dataset_posterior_entries)
            if posterior_seedmean_series:
                p = plots_dir / "posterior_U_compare_seedmean_raw.html"
                g_grid = np.arange(int(cfg.poster_min), int(cfg.poster_max) + 1, dtype=float)
                truth_y = np.asarray(true_utility(g_grid), dtype=float)

                plot_posterior_U_overlay_plotly(
                    posterior_seedmean_series,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=p,
                    title="",
                    scale_mode="raw",
                    yaxis_title="Utility",
                    show_hdi_band=True,
                    band_label="+/- 1 SD across seed-level posterior means",
                    truth_y=truth_y,
                    truth_label="Ground truth",
                    truth_color="black",
                    truth_dash="solid",
                    truth_width=4,
                )
                dataset_combined["compare_posterior_U_seedmean_raw_html"] = str(p)
                p_png = plots_dir / "thesis_posterior_U_compare_seedmean_raw.png"
                if plot_posterior_U_overlay_png(
                    posterior_seedmean_series,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_png=p_png,
                    scale_mode="raw",
                    yaxis_title="Utility",
                    band_alpha=0.20,
                    band_source="sd",
                    show_hdi_band=True,
                    truth_y=truth_y,
                    truth_label="Ground truth",
                    truth_color="black",
                    truth_dash="solid",
                    truth_width=2.6,
                    line_width=1.45,
                    legend_fontsize=9.0,
                    legend_ncol=3,
                    figsize=(10.5, 4.8),
                ):
                    dataset_combined["thesis_posterior_U_seedmean_raw_png"] = str(p_png)
                p_individual_dir = plots_dir / "thesis_posterior_U_by_method"
                individual_pngs = plot_posterior_U_individual_pngs(
                    posterior_seedmean_series,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_dir=p_individual_dir,
                    scale_mode="raw",
                    yaxis_title="Utility",
                    band_alpha=0.20,
                    band_source="sd",
                    show_hdi_band=True,
                    truth_y=truth_y,
                    truth_label="Ground truth",
                    truth_color="black",
                    truth_dash="solid",
                    truth_width=2.6,
                    line_width=1.9,
                    legend_fontsize=9.0,
                )
                if individual_pngs:
                    dataset_combined["thesis_posterior_U_by_method_png_dir"] = str(p_individual_dir)

        for r in dataset_rows:
            r.update(dataset_combined)

        dataset_variant_rows = []
        for v in variants:
            vr = {
                "variant_label": _variant_label_from_spec(v),
                "infer_model_name": str(v.get("model_name", gen_model_name)),
                "likelihood": str(v.get("likelihood", cfg.likelihood)),
                "prior": str(v.get("prior", cfg.prior)),
            }
            if "alpha_fixed" in v:
                vr["alpha_fixed"] = float(v["alpha_fixed"])
            dataset_variant_rows.append(vr)
        dataset_alpha_values = sorted(
            {
                float(vr["alpha_fixed"])
                for vr in dataset_variant_rows
                if "alpha_fixed" in vr
            }
        )

        dataset_suite_summary = {
            "dataset_suite": dataset_suite,
            "dataset_label": dataset_label,
            "utility_kind": utility_kind,
            "train_file": str(train_path),
            "stem": train_path.stem,
            "job": _jsonable_job(j),
            "gamma_fixed": float(gamma_fixed),
            "alpha_fixed": (
                float(dataset_alpha_values[0])
                if len(dataset_alpha_values) == 1
                else (float(alpha_fixed) if not dataset_alpha_values else None)
            ),
            "alpha_values": dataset_alpha_values,
            "variant_set": variant_set,
            "seed_ids": [int(s) for s in seed_ids],
            "base_mcmc_seed": int(base_mcmc_seed),
            "combined_artifacts": dataset_combined,
            "variants": dataset_variant_rows,
        }

        (dataset_run_dir / "suite_summary.json").write_text(
            json.dumps(dataset_suite_summary, indent=2),
            encoding="utf-8",
        )

    # Re-scan the run root so appending seed ids across multiple invocations
    # still produces aggregate artifacts over every completed seed on disk.
    aggregate_rows = _aggregate_existing_seed_outputs(
        source_roots=[run_root],
        output_root=run_root,
        utility_mode="both",
    )
    if aggregate_rows:
        rows = aggregate_rows

    df = _write_combined_run_summary(rows, run_root)

    print(df)
    print(f"\nAll results saved under: {run_root}")


if __name__ == "__main__":
    main()
