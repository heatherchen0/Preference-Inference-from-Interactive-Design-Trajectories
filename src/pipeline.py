from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import ExperimentConfig, to_dict, validate_experiment_config
from .io_utils import load_jsonl
from .model import (
    build_model2_discounted,
    build_model4_endpoint_discounted_boltzmann,
    build_model4_endpoint_discounted_boltzmann_infer_alpha,
    build_model4_endpoint_discounted_boltzmann_2d,
    build_birl_baseline_discounted_boltzmann,
    build_birl_baseline_discounted_boltzmann_2d,
    build_pbo_pairwise_preference_model,
)
from .preprocess import (
    prepare_inverse_data,
    prepare_pbo_start_end_data,
    prepare_pbo_transition_data,
)
from .sampling import sample_model
from .prefix_fit import get_or_fit_prefix_summaries
from .plots import (
    plot_utility,
    plot_interactive_prefix_posterior_states_plotly,
    plot_prefix_mean_heatmap_states_plotly,
    plot_prefix_metrics_learning_curves_plotly,
    plot_posterior_U_overlay_plotly,
    plot_2d_posterior_heatmaps_plotly,
    plot_2d_prefix_heatmap_slider_plotly,
    plot_contrast_profile_plotly,
)
from .metrics import (
    compute_raw_mse_from_idata,
    compute_curve_error_metrics,
    write_per_state_utility_table,
    compute_pairwise_accuracy,
    compute_posterior_pairwise_accuracy_from_idata,
    summarize_posterior_var,
    summarize_soft_copeland_from_idata,
    compute_recommendation_metrics,
    compute_gaussian_log_score_metrics,
    compute_contrast_profile_metrics,
    build_contrast_profile_table,
    write_contrast_profile_table,
)
from utility import get_utility
from .prefix_analysis import (
    materialize_prefix_summaries,
    write_alpha_prefix_summary_csv,
    write_prefix_metrics_csv,
    write_prefix_per_state_tables,
    write_prefix_contrast_profile_tables,
)
from .elpd import compute_elpd_for_df
from .posterior_archive import (
    save_alpha_draw_archive,
    save_u_draw_archive,
    write_archive_manifest,
    summarize_idata_diagnostics,
    write_json,
    write_rows_csv,
)
from .state_space import (
    action_order,
    normalize_state_space,
    normalized_gp_inputs,
    poster_to_coord_2d,
)

def _ensure_run_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _optional_plotly():
    try:
        import plotly.express as px

        return True, px
    except Exception:
        return False, None


def _state_space(cfg: ExperimentConfig) -> str:
    return normalize_state_space(getattr(cfg, "state_space", "1d"))


def _grid_k(cfg: ExperimentConfig) -> int | None:
    return int(getattr(cfg, "grid_k", 10)) if _state_space(cfg) == "2d" else None


def _preferred_contrast(cfg: ExperimentConfig):
    meta = getattr(cfg, "utility_metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("preferred_contrast", "c_star"):
            if key in meta and meta[key] is not None:
                return float(meta[key])
    return None


def _summarize_actions(action_idx: np.ndarray, *, state_space: str = "1d") -> dict:
    action_idx = np.asarray(action_idx, dtype=int)
    return {
        name: int(np.sum(action_idx == idx))
        for idx, name in enumerate(action_order(state_space))
    }


def _utility_kind(cfg: ExperimentConfig) -> str:
    return str(getattr(cfg, "utility_kind", "smooth")).strip().lower()


def _ground_truth_utility_and_metadata(cfg: ExperimentConfig):
    kind = _utility_kind(cfg)
    if _state_space(cfg) == "2d":
        if kind != "smooth_contrast_interaction":
            raise ValueError(f"Unsupported 2D utility_kind={kind!r}.")
        from poster2d.utility import (
            smooth_contrast_interaction_default_params,
            smooth_contrast_interaction_utility,
        )

        k = int(_grid_k(cfg))
        params = smooth_contrast_interaction_default_params(k=k)

        def utility_fn(posters):
            return smooth_contrast_interaction_utility(posters, k=k)

        return utility_fn, {"utility_params": params, "preferred_contrast": params.get("preferred_contrast")}

    return get_utility(kind), {}


def _truth_utility_grid_summary(
    utility_fn,
    *,
    poster_min: int,
    poster_max: int,
    state_space: str = "1d",
    grid_k: int | None = None,
) -> dict:
    posters = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
    values = np.asarray(utility_fn(posters), dtype=float).reshape(-1)

    argmax_idx = int(np.argmax(values))
    argmin_idx = int(np.argmin(values))

    # Simple local maxima diagnostic.
    local_maxima = []
    if values.size >= 3:
        mid = values[1:-1]
        left = values[:-2]
        right = values[2:]
        mask = (mid >= left) & (mid >= right)
        local_maxima = posters[1:-1][mask].astype(int).tolist()

    order = np.argsort(values)[::-1]
    top5 = [
        {
            "poster": int(posters[i]),
            "U": float(values[i]),
        }
        for i in order[:5]
    ]

    out = {
        "poster_min": int(poster_min),
        "poster_max": int(poster_max),
        "U_min": float(np.min(values)),
        "U_max": float(np.max(values)),
        "argmin_poster": int(posters[argmin_idx]),
        "argmax_poster": int(posters[argmax_idx]),
        "top5_posters": top5,
        "local_maxima_posters": local_maxima,
        "num_local_maxima": int(len(local_maxima)),
    }
    if normalize_state_space(state_space) == "2d":
        k = int(grid_k)
        h_star, b_star = poster_to_coord_2d(int(posters[argmax_idx]), grid_k=k)
        h_min, b_min = poster_to_coord_2d(int(posters[argmin_idx]), grid_k=k)
        out.update(
            {
                "grid_k": k,
                "argmax_headline_gray": int(h_star),
                "argmax_background_gray": int(b_star),
                "argmax_contrast": int(abs(h_star - b_star)),
                "argmin_headline_gray": int(h_min),
                "argmin_background_gray": int(b_min),
                "argmin_contrast": int(abs(h_min - b_min)),
            }
        )
    return out
    
    
def _model_label(cfg: ExperimentConfig) -> str:
    return str(getattr(cfg, "model_name", "model2_discounted"))


_PBO_START_END_MODELS = ("pbo_baseline", "pbo_baseline_start_end_preference")
_PBO_TRANSITION_MODELS = ("pbo_baseline_transition_preference",)
_PBO_MODELS = _PBO_START_END_MODELS + _PBO_TRANSITION_MODELS


def _is_pbo_model_name(model_name: str) -> bool:
    return str(model_name).lower() in _PBO_MODELS


def _infers_alpha_model(model_name: str) -> bool:
    return str(model_name).lower() == "model4_endpoint_discounted_boltzmann_infer_alpha"


def _alpha_mode(model_name: str) -> str:
    return "inferred" if _infers_alpha_model(model_name) else "fixed"


def _pbo_duel_label(model_name: str) -> str:
    name = str(model_name).lower()
    if name in _PBO_TRANSITION_MODELS:
        return "transition"
    if name in _PBO_START_END_MODELS:
        return "start-end"
    raise ValueError(f"Unknown PBO model_name={model_name!r}")


def _pbo_likelihood_label(model_name: str) -> str:
    return (
        "transition_logistic"
        if _pbo_duel_label(model_name) == "transition"
        else "start_end_logistic"
    )


def _prepare_pbo_data_for_model(
    model_name: str,
    data,
    *,
    poster_min: int,
    poster_max: int,
    state_space: str = "1d",
    grid_k: int | None = None,
):
    if _pbo_duel_label(model_name) == "transition":
        return prepare_pbo_transition_data(
            data,
            poster_min=poster_min,
            poster_max=poster_max,
            drop_ties=True,
            state_space=state_space,
            grid_k=grid_k,
        )
    return prepare_pbo_start_end_data(
        data,
        poster_min=poster_min,
        poster_max=poster_max,
        drop_ties=True,
        state_space=state_space,
        grid_k=grid_k,
    )


def _plot_title(
    cfg: ExperimentConfig,
    *,
    gamma_fixed: float,
    prior: str,
    likelihood: str,
    include_truth: bool = True,
) -> str:
    model_label = _model_label(cfg)
    parts = []
    if include_truth:
        parts.append(f"truth={_utility_kind(cfg)}")
    parts.extend([
        f"prior={prior}",
        f"likelihood={likelihood}",
        f"gamma={float(gamma_fixed)}",
    ])

    model_name = str(getattr(cfg, "model_name", "")).lower()

    if model_name in ("model4_endpoint_discounted_boltzmann", "birl_baseline", "birl_baseline_discounted_boltzmann"):
        parts.append(f"alpha={float(getattr(cfg, 'alpha_fixed', 5.0))}")

    if _infers_alpha_model(model_name):
        parts.append(
            "alpha~LogNormal"
            f"(median={float(getattr(cfg, 'alpha_prior_median', 5.0))}, "
            f"log_sd={float(getattr(cfg, 'alpha_prior_log_sd', 0.75))})"
        )

    if model_name in ("birl_baseline", "birl_baseline_discounted_boltzmann"):
        parts.append(f"export={str(getattr(cfg, 'birl_export_mode'))}")
        parts.append(f"reward_scale={str(getattr(cfg, 'birl_reward_scale'))}")
        parts.append(f"value_iters={int(getattr(cfg, 'birl_value_iters', 1000))}")

    if _is_pbo_model_name(model_name):
        parts.append(f"duels={_pbo_duel_label(model_name)}")

    return f"Forward model: {model_label}<br>Inference: " + ", ".join(parts)


def _posterior_U_summary(idata, hdi_prob: float = 0.94):
    import arviz as az

    if "U" not in idata.posterior:
        raise KeyError("U not found in posterior; expected Deterministic('U', ...) in model.")

    U = idata.posterior["U"]
    mean = U.mean(dim=("chain", "draw")).values

    hdi_obj = az.hdi(U, hdi_prob=hdi_prob)

    if hasattr(hdi_obj, "data_vars"):
        if "U" in hdi_obj.data_vars:
            hdi = hdi_obj["U"]
        else:
            first_var = next(iter(hdi_obj.data_vars))
            hdi = hdi_obj[first_var]
    else:
        hdi = hdi_obj

    if hasattr(hdi, "sel") and "hdi" in getattr(hdi, "coords", {}):
        low = hdi.sel(hdi="lower").values
        high = hdi.sel(hdi="higher").values
        return np.asarray(mean), np.asarray(low), np.asarray(high)

    hdi_np = np.asarray(getattr(hdi, "values", hdi))
    if hdi_np.ndim >= 2 and hdi_np.shape[-1] == 2:
        low = hdi_np[..., 0]
        high = hdi_np[..., 1]
        return np.asarray(mean), np.asarray(low), np.asarray(high)

    raise TypeError(
        "Could not interpret HDI output from arviz. "
        f"type={type(hdi_obj)} extracted_type={type(hdi)} "
        f"dims={getattr(hdi, 'dims', None)} coords={list(getattr(hdi, 'coords', {}).keys())}"
    )


def _posterior_alpha_summary(
    idata,
    *,
    hdi_prob: float,
    alpha_prior_median: float,
    alpha_prior_log_sd: float,
) -> dict | None:
    if not hasattr(idata, "posterior") or "alpha" not in idata.posterior:
        return None
    out = summarize_posterior_var(
        idata,
        var_name="alpha",
        hdi_prob=float(hdi_prob),
        normalize_draws=False,
    )
    def _scalar(x):
        return float(np.asarray(x, dtype=float).reshape(-1)[0])
    draws = np.asarray(idata.posterior["alpha"].values, dtype=float).reshape(-1)

    return {
        "mean": _scalar(out["mean"]),
        "median": float(np.nanmedian(draws)),
        "sd": _scalar(out["sd"]),
        "q05": float(np.nanquantile(draws, 0.05)),
        "q95": float(np.nanquantile(draws, 0.95)),
        "low": _scalar(out["low"]),
        "high": _scalar(out["high"]),
        "hdi_prob": float(out["hdi_prob"]),
        "var_name": "alpha",
        "prior": "LogNormal",
        "prior_median": float(alpha_prior_median),
        "prior_log_sd": float(alpha_prior_log_sd),
    }


def _plot_U_band_html(
    mean,
    low,
    high,
    *,
    poster_min: int,
    out_html: Path,
    gamma_fixed: float,
    hdi_prob: float,
    title: str,
    yaxis_title: str = "Utility U(g)",
    truth_y=None,
    truth_label: str = "Ground truth U(g)",
) -> bool:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)

    if truth_y is not None:
        truth_y = np.asarray(truth_y, dtype=float).reshape(-1)
        if truth_y.shape[0] != mean.shape[0]:
            raise ValueError(
                f"truth_y has length {truth_y.shape[0]} but posterior mean has length {mean.shape[0]}."
            )

    try:
        plot_posterior_U_overlay_plotly(
            [
                {
                    "label": "Posterior mean U",
                    "mean": mean,
                    "low": low,
                    "high": high,
                    "already_normalized": "min-max" in str(yaxis_title).lower(),
                    "normalization_method": (
                        "posterior_draw_minmax"
                        if "min-max" in str(yaxis_title).lower()
                        else ""
                    ),
                }
            ],
            poster_min=int(poster_min),
            poster_max=int(poster_min) + int(mean.shape[0]) - 1,
            out_path_html=out_html,
            title=title,
            scale_mode=(
                "posterior_draw_minmax"
                if "min-max" in str(yaxis_title).lower()
                else "raw"
            ),
            yaxis_title=str(yaxis_title),
            hdi_prob=float(hdi_prob),
            show_hdi_band=True,
            truth_y=truth_y,
            truth_label=str(truth_label),
            truth_color="black",
            truth_dash="solid",
            truth_width=3,
        )
        return True
    except ImportError:
        return False


def _try_rank_plots(idata, out_dir: Path) -> bool:
    """
    Optional rank plots if ArviZ + Plotly are available.
    """
    ok, _px = _optional_plotly()
    if not ok:
        return False

    try:
        import arviz as az
    except Exception:
        return False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    varnames = []
    for v in ("rw_sigma", "f_raw", "U"):
        if v in idata.posterior:
            varnames.append(v)

    if not varnames:
        return False

    for v in varnames:
        try:
            fig = az.plot_rank(idata, var_names=[v], kind="bars")
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig(out_dir / f"rank_{v}.png", dpi=150)
            plt.close()
        except Exception:
            return False

    return True


def run_experiment(
    *,
    cfg: ExperimentConfig,
    data_path: str | Path,
    run_dir: str | Path,
    gamma_fixed: float = 0.9,
    hdi_prob: float = 0.94,
    trajectory_order=None,
    trajectory_permutation_seed: int | None = None,
) -> dict:
    run_dir = _ensure_run_dir(run_dir)
    data_path = Path(data_path)
    
    validate_experiment_config(cfg)

    utility_kind = _utility_kind(cfg)
    state_space = _state_space(cfg)
    grid_k = _grid_k(cfg)
    preferred_contrast = _preferred_contrast(cfg)
    ground_truth_utility, utility_metadata = _ground_truth_utility_and_metadata(cfg)
    if preferred_contrast is None:
        preferred_contrast = utility_metadata.get("preferred_contrast")

    truth_utility_summary = _truth_utility_grid_summary(
        ground_truth_utility,
        poster_min=int(cfg.poster_min),
        poster_max=int(cfg.poster_max),
        state_space=state_space,
        grid_k=grid_k,
    )

    rows = load_jsonl(data_path)

    state_idx, action_idx, num_states = prepare_inverse_data(
        rows,
        poster_min=cfg.poster_min,
        poster_max=cfg.poster_max,
        state_space=state_space,
    )
    prior = str(cfg.prior).lower()
    gp_inputs = (
        normalized_gp_inputs(
            state_space=state_space,
            num_states=num_states,
            grid_k=grid_k,
        )
        if prior
        in {
            "gp",
            "gp_rbf",
            "rbf",
            "logistic_gp",
            "logistic_gp_rbf",
            "gp_uniform",
            "gp_uniform_rbf",
            "gp_copula",
            "gp_copula_rbf",
            "copula_gp",
            "copula_gp_rbf",
        }
        else None
    )

    model_name = str(getattr(cfg, "model_name", "model2_discounted")).lower()
    is_pbo_model = _is_pbo_model_name(model_name)
    likelihood = str(cfg.likelihood).lower()
    pbo_stats = None
    if model_name == "model2_discounted":
        model = build_model2_discounted(
            state_idx=state_idx,
            action_idx=action_idx,
            num_states=num_states,
            gamma_fixed=float(gamma_fixed),
            beta_fixed=float(cfg.beta_fixed),
            likelihood=likelihood,
            eps_fixed=0.01,
            infer_eps=False,
            prior=prior,
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
        )
    elif model_name == "model4_endpoint_discounted_boltzmann":
        if state_space == "2d":
            model = build_model4_endpoint_discounted_boltzmann_2d(
                state_idx=state_idx,
                action_idx=action_idx,
                num_states=num_states,
                grid_k=int(grid_k),
                gamma_fixed=float(gamma_fixed),
                alpha_fixed=float(cfg.alpha_fixed),
                prior=prior,
                gp_eta_sd=float(cfg.gp_eta_sd),
                gp_ls_sd=float(cfg.gp_ls_sd),
                gp_inputs=gp_inputs,
            )
        else:
            model = build_model4_endpoint_discounted_boltzmann(
                state_idx=state_idx,
                action_idx=action_idx,
                num_states=num_states,
                gamma_fixed=float(gamma_fixed),
                alpha_fixed=float(cfg.alpha_fixed),
                prior=prior,
                gp_eta_sd=float(cfg.gp_eta_sd),
                gp_ls_sd=float(cfg.gp_ls_sd),
                gp_inputs=gp_inputs,
            )
        likelihood = "boltzmann_qstar"
    elif model_name == "model4_endpoint_discounted_boltzmann_infer_alpha":
        if state_space != "1d":
            raise ValueError(
                "model4_endpoint_discounted_boltzmann_infer_alpha is v1 1D-only; "
                f"got state_space={state_space!r}."
            )
        model = build_model4_endpoint_discounted_boltzmann_infer_alpha(
            state_idx=state_idx,
            action_idx=action_idx,
            num_states=num_states,
            gamma_fixed=float(gamma_fixed),
            alpha_prior_median=float(cfg.alpha_prior_median),
            alpha_prior_log_sd=float(cfg.alpha_prior_log_sd),
            prior=prior,
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
            gp_inputs=gp_inputs,
        )
        likelihood = "boltzmann_qstar"
    elif model_name in ("birl_baseline", "birl_baseline_discounted_boltzmann"):
        if state_space == "2d":
            model = build_birl_baseline_discounted_boltzmann_2d(
                state_idx=state_idx,
                action_idx=action_idx,
                num_states=num_states,
                grid_k=int(grid_k),
                gamma_fixed=float(gamma_fixed),
                alpha_fixed=float(cfg.alpha_fixed),
                prior=prior,
                export_mode=str(getattr(cfg, "birl_export_mode")),
                reward_scale=str(getattr(cfg, "birl_reward_scale")),
                value_iters=int(getattr(cfg, "birl_value_iters", 1000)),
                gp_eta_sd=float(cfg.gp_eta_sd),
                gp_ls_sd=float(cfg.gp_ls_sd),
                gp_inputs=gp_inputs,
            )
        else:
            model = build_birl_baseline_discounted_boltzmann(
                state_idx=state_idx,
                action_idx=action_idx,
                num_states=num_states,
                gamma_fixed=float(gamma_fixed),
                alpha_fixed=float(cfg.alpha_fixed),
                prior=prior,
                export_mode=str(getattr(cfg, "birl_export_mode")),
                reward_scale=str(getattr(cfg, "birl_reward_scale")),
                value_iters=int(getattr(cfg, "birl_value_iters", 1000)),
                gp_eta_sd=float(cfg.gp_eta_sd),
                gp_ls_sd=float(cfg.gp_ls_sd),
                gp_inputs=gp_inputs,
            )
        likelihood = "boltzmann_qstar"
    elif is_pbo_model:
        start_idx, end_idx, num_states_pbo, pbo_stats, _pbo_cmp = _prepare_pbo_data_for_model(
            model_name,
            rows,
            poster_min=int(cfg.poster_min),
            poster_max=int(cfg.poster_max),
            state_space=state_space,
            grid_k=grid_k,
        )
        if int(num_states_pbo) != int(num_states):
            raise RuntimeError(
                f"Unexpected num_states mismatch for PBO: {num_states_pbo} vs {num_states}"
            )
        model = build_pbo_pairwise_preference_model(
            start_idx=start_idx,
            end_idx=end_idx,
            num_states=num_states,
            prior=prior,
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
            gp_inputs=gp_inputs,
        )
        likelihood = _pbo_likelihood_label(model_name)
    else:
        raise ValueError(f"Unknown cfg.model_name={cfg.model_name!r}")
    
    idata = sample_model(model, cfg.mcmc_main)
    
    # -------------------------
    # ELPD evaluation (end-only)
    # -------------------------
    elpd_outputs = {}
    eval_cfg = getattr(cfg, "eval", None)

    def _write_elpd(res, tag: str):
        out_dir = run_dir / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_traj_path = out_dir / f"elpd_{tag}_per_traj.csv"
        by_start_path = out_dir / f"elpd_{tag}_by_start.csv"

        res.per_traj.to_csv(per_traj_path, index=False)
        res.by_start.to_csv(by_start_path, index=False)

        return {
            "summary": res.summary,
            "per_traj_csv": str(per_traj_path),
            "by_start_csv": str(by_start_path),
        }

    df_test_seen = None
    if (
        eval_cfg
        and bool(eval_cfg.enabled)
        and str(eval_cfg.test_seen_path).strip()
        and not is_pbo_model
        and state_space == "1d"
    ):
        test_seen_path = Path(str(eval_cfg.test_seen_path))
        rows_seen = load_jsonl(test_seen_path)
        df_test_seen = pd.DataFrame(rows_seen)

        res_seen = compute_elpd_for_df(
            idata=idata,
            df_eval=df_test_seen,
            dataset_name="main_seen_test",
            poster_min=int(cfg.poster_min),
            poster_max=int(cfg.poster_max),
            model_name=str(cfg.model_name),
            gamma_fixed=float(gamma_fixed),
            beta_fixed=float(cfg.beta_fixed),
            alpha_fixed=float(getattr(cfg, "alpha_fixed", 5.0)),
            alpha_prior_median=float(getattr(cfg, "alpha_prior_median", 5.0)),
            alpha_prior_log_sd=float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
            likelihood=str(likelihood),
            prior=str(prior),
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
            obs_var_name=str(getattr(eval_cfg, "obs_var_name", "obs")),
        )
        elpd_outputs["main_seen"] = _write_elpd(res_seen, tag="main_seen")

    if (
        eval_cfg
        and bool(eval_cfg.enabled)
        and str(eval_cfg.test_unseen_path).strip()
        and not is_pbo_model
        and state_space == "1d"
    ):
        test_unseen_path = Path(str(eval_cfg.test_unseen_path))
        rows_unseen = load_jsonl(test_unseen_path)
        df_test_unseen = pd.DataFrame(rows_unseen)

        res_unseen = compute_elpd_for_df(
            idata=idata,
            df_eval=df_test_unseen,
            dataset_name="main_unseen_test",
            poster_min=int(cfg.poster_min),
            poster_max=int(cfg.poster_max),
            model_name=str(cfg.model_name),
            gamma_fixed=float(gamma_fixed),
            beta_fixed=float(cfg.beta_fixed),
            alpha_fixed=float(getattr(cfg, "alpha_fixed", 5.0)),
            alpha_prior_median=float(getattr(cfg, "alpha_prior_median", 5.0)),
            alpha_prior_log_sd=float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
            likelihood=str(likelihood),
            prior=str(prior),
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
            obs_var_name=str(getattr(eval_cfg, "obs_var_name", "obs")),
        )
        elpd_outputs["main_unseen"] = _write_elpd(res_unseen, tag="main_unseen")

    U_raw_summary = summarize_posterior_var(
        idata,
        var_name="U",
        hdi_prob=float(hdi_prob),
        normalize_draws=False,
    )
    U_norm_summary = summarize_posterior_var(
        idata,
        var_name="U",
        hdi_prob=float(hdi_prob),
        normalize_draws=True,
    )
    alpha_summary = _posterior_alpha_summary(
        idata,
        hdi_prob=float(hdi_prob),
        alpha_prior_median=float(getattr(cfg, "alpha_prior_median", 5.0)),
        alpha_prior_log_sd=float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
    )

    mean = np.asarray(U_raw_summary["mean"], dtype=float)
    low = np.asarray(U_raw_summary["low"], dtype=float)
    high = np.asarray(U_raw_summary["high"], dtype=float)

    mean_norm = np.asarray(U_norm_summary["mean"], dtype=float)
    low_norm = np.asarray(U_norm_summary["low"], dtype=float)
    high_norm = np.asarray(U_norm_summary["high"], dtype=float)
    
    raw_mse_out = compute_raw_mse_from_idata(
        idata,
        utility_fn=ground_truth_utility,
        poster_min=cfg.poster_min,
        poster_max=cfg.poster_max,
        var_name="U",
    )
    
    curve_metrics_raw = compute_curve_error_metrics(
        posters=raw_mse_out.posters,
        U_true=raw_mse_out.U_true,
        U_post_mean=raw_mse_out.U_post_mean,
        U_low=np.asarray(low),
        U_high=np.asarray(high),
    )
    log_score_raw = compute_gaussian_log_score_metrics(
        U_true=raw_mse_out.U_true,
        U_post_mean=raw_mse_out.U_post_mean,
        U_post_sd=raw_mse_out.U_post_sd,
        variance_floor=1e-6,
    )
    log_score_raw = {
        **log_score_raw,
        "gaussian_nlpd_raw": log_score_raw["gaussian_nlpd"],
        "gaussian_log_score_raw": log_score_raw["gaussian_log_score"],
        "gaussian_nlpd_var_floor": log_score_raw["variance_floor"],
        "gaussian_nlpd_num_floored": log_score_raw["num_floored"],
    }

    ranking_metrics_raw = compute_pairwise_accuracy(
        U_true=raw_mse_out.U_true,
        U_pred=raw_mse_out.U_post_mean,
        eps=1e-6,
    )

    ranking_metrics_posterior = compute_posterior_pairwise_accuracy_from_idata(
        idata,
        U_true=raw_mse_out.U_true,
        var_name="U",
        eps_true=1e-6,
        eps_pred=0.0,
    )

    curve_metrics_norm = compute_curve_error_metrics(
        posters=raw_mse_out.posters,
        U_true=raw_mse_out.U_true,
        U_post_mean=mean_norm,
        U_low=low_norm,
        U_high=high_norm,
    )

    ranking_metrics_norm = compute_pairwise_accuracy(
        U_true=raw_mse_out.U_true,
        U_pred=mean_norm,
        eps=1e-6,
    )

    soft_copeland_summary = summarize_soft_copeland_from_idata(
        idata,
        var_name="U",
        hdi_prob=float(hdi_prob),
    )
    recommendation_score = np.asarray(soft_copeland_summary["mean"], dtype=float)
    recommendation_score_name = "soft_copeland"

    recommendation_metrics = compute_recommendation_metrics(
        posters=raw_mse_out.posters,
        U_true=raw_mse_out.U_true,
        recommendation_score=recommendation_score,
        score_name=recommendation_score_name,
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )

    archive_cfg = getattr(cfg, "posterior_archive", None)
    save_u_draws = bool(getattr(archive_cfg, "save_u_draws", False))
    save_alpha_draws = bool(save_u_draws and alpha_summary is not None)
    save_archive_diagnostics = bool(getattr(archive_cfg, "save_diagnostics", False))
    save_full_idata = bool(getattr(archive_cfg, "save_full_idata", False))
    archive_dtype = str(getattr(archive_cfg, "dtype", "float32"))
    archive_dir = run_dir / "posterior_archive"
    archive_entries: list[dict] = []
    archive_outputs: dict[str, object] = {
        "enabled": bool(save_u_draws or save_alpha_draws or save_archive_diagnostics or save_full_idata),
        "save_u_draws": save_u_draws,
        "save_alpha_draws": save_alpha_draws,
        "save_diagnostics": save_archive_diagnostics,
        "save_full_idata": save_full_idata,
        "dtype": archive_dtype,
    }
    main_u_archive_path = None
    main_alpha_archive_path = None
    main_idata_archive_path = None
    main_archive_info = None
    main_alpha_archive_info = None

    archive_base_metadata = {
        "data_path": str(data_path),
        "utility_kind": str(utility_kind),
        "utility_metadata": utility_metadata,
        "state_space": state_space,
        "grid_k": grid_k,
        "action_order": list(action_order(state_space)),
        "preferred_contrast": (
            None if preferred_contrast is None else float(preferred_contrast)
        ),
        "coordinate_mapping": (
            "poster(h,b)=(h-1)*K+b; state_idx=poster-1"
            if state_space == "2d"
            else "state_idx=poster-poster_min"
        ),
        "model_name": str(cfg.model_name),
        "likelihood": str(likelihood),
        "prior": str(prior),
        "gamma_fixed": float(gamma_fixed),
        "alpha_fixed": float(getattr(cfg, "alpha_fixed", 5.0)),
        "alpha_mode": _alpha_mode(str(cfg.model_name)),
        "alpha_prior_median": float(getattr(cfg, "alpha_prior_median", 5.0)),
        "alpha_prior_log_sd": float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
        "beta_fixed": float(getattr(cfg, "beta_fixed", 20.0)),
        "poster_min": int(cfg.poster_min),
        "poster_max": int(cfg.poster_max),
        "mcmc_main_random_seed": int(cfg.mcmc_main.random_seed),
        "trajectory_permutation_seed": (
            int(trajectory_permutation_seed)
            if trajectory_permutation_seed is not None
            else None
        ),
    }

    if save_u_draws:
        main_archive_info = save_u_draw_archive(
            idata=idata,
            out_path=archive_dir / "main_U_draws.npz",
            posters=raw_mse_out.posters,
            U_true=raw_mse_out.U_true,
            metadata={**archive_base_metadata, "scope": "main"},
            dtype=archive_dtype,
        )
        main_u_archive_path = str(main_archive_info["path"])
        archive_entries.append({"scope": "main", "kind": "u_draws", **main_archive_info})
        archive_outputs["main_u_draws_npz"] = main_u_archive_path

    if save_alpha_draws:
        main_alpha_archive_info = save_alpha_draw_archive(
            idata=idata,
            out_path=archive_dir / "alpha_draws" / "main_alpha_draws.npz",
            metadata={**archive_base_metadata, "scope": "main"},
            dtype=archive_dtype,
        )
        main_alpha_archive_path = str(main_alpha_archive_info["path"])
        archive_entries.append({"scope": "main", "kind": "alpha_draws", **main_alpha_archive_info})
        archive_outputs["main_alpha_draws_npz"] = main_alpha_archive_path

    if save_full_idata:
        main_idata_path = archive_dir / "main_idata.nc"
        main_idata_path.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(main_idata_path)
        main_idata_archive_path = str(main_idata_path)
        archive_outputs["main_idata_netcdf"] = main_idata_archive_path
        archive_entries.append(
            {
                "scope": "main",
                "kind": "full_idata",
                "path": main_idata_archive_path,
            }
        )

    if save_archive_diagnostics:
        main_alpha_diag = None
        try:
            main_diag, main_diag_per_state = summarize_idata_diagnostics(
                idata,
                var_name="U",
                posters=raw_mse_out.posters,
            )
            main_diag = {**archive_base_metadata, **main_diag, "scope": "main"}
            main_diag_path = write_json(main_diag, archive_dir / "diagnostics" / "main_diagnostics.json")
            main_diag_state_path = write_rows_csv(
                main_diag_per_state,
                archive_dir / "diagnostics" / "main_U_diagnostics_per_state.csv",
            )
            archive_outputs["main_diagnostics_json"] = main_diag_path
            archive_outputs["main_u_diagnostics_per_state_csv"] = main_diag_state_path
        except Exception as e:
            main_diag = {
                **archive_base_metadata,
                "scope": "main",
                "diagnostics_error": str(e),
            }
            main_diag_path = write_json(main_diag, archive_dir / "diagnostics" / "main_diagnostics.json")
            archive_outputs["main_diagnostics_json"] = main_diag_path
        if alpha_summary is not None:
            try:
                main_alpha_diag, _main_alpha_diag_rows = summarize_idata_diagnostics(
                    idata,
                    var_name="alpha",
                )
                main_alpha_diag = {
                    **archive_base_metadata,
                    **main_alpha_diag,
                    "scope": "main",
                }
            except Exception as e:
                main_alpha_diag = {
                    **archive_base_metadata,
                    "scope": "main",
                    "var_name": "alpha",
                    "diagnostics_error": str(e),
                }
            main_alpha_diag_path = write_json(
                main_alpha_diag,
                archive_dir / "diagnostics" / "main_alpha_diagnostics.json",
            )
            archive_outputs["main_alpha_diagnostics_json"] = main_alpha_diag_path
    else:
        main_diag = None
        main_alpha_diag = None

    curve_metrics_norm = dict(curve_metrics_norm)
    curve_metrics_norm.update(
        {
            "pairwise_acc_comp": ranking_metrics_norm["pairwise_acc_comp"],
            "pairwise_acc_strict": ranking_metrics_norm["pairwise_acc_strict"],
            "pairwise_num_pairs_comparable": ranking_metrics_norm["num_pairs_comparable"],
            "pairwise_num_correct_comparable": ranking_metrics_norm["num_correct_comparable"],
            "pairwise_eps": ranking_metrics_norm["eps"],
            "posterior_pairwise_acc_comp": ranking_metrics_posterior["posterior_pairwise_acc_comp"],
            "posterior_pairwise_acc_strict": ranking_metrics_posterior["posterior_pairwise_acc_strict"],
            "posterior_pairwise_num_pairs_comparable": ranking_metrics_posterior["num_pairs_comparable"],
            "posterior_pairwise_expected_correct_comparable": ranking_metrics_posterior[
                "posterior_pairwise_expected_correct_comparable"
            ],
            "posterior_pairwise_num_draws": ranking_metrics_posterior["n_draws"],
            "posterior_pairwise_eps_true": ranking_metrics_posterior["eps_true"],
            "posterior_pairwise_eps_pred": ranking_metrics_posterior["eps_pred"],
            "posterior_pairwise_acc_draw_sd": ranking_metrics_posterior[
                "posterior_pairwise_acc_draw_sd"
            ],
            "normalization_method": "posterior_draw_minmax",
            "g_hat": recommendation_metrics["g_hat"],
            "U_true_at_g_hat": recommendation_metrics["U_true_at_g_hat"],
            "simple_regret": recommendation_metrics["simple_regret"],
            "g_star": recommendation_metrics["g_star"],
            "U_true_star": recommendation_metrics["U_true_star"],
        }
    )
    if state_space == "2d":
        curve_metrics_norm.update(
            compute_contrast_profile_metrics(
                U_true=raw_mse_out.U_true,
                U_pred=mean_norm,
                grid_k=int(grid_k),
            )
        )
        curve_metrics_norm.update(
            {
                "h_hat": recommendation_metrics.get("h_hat"),
                "b_hat": recommendation_metrics.get("b_hat"),
                "c_hat": recommendation_metrics.get("c_hat"),
                "h_star": recommendation_metrics.get("h_star"),
                "b_star": recommendation_metrics.get("b_star"),
                "c_star": recommendation_metrics.get("c_star"),
                "contrast_error": recommendation_metrics.get("contrast_error"),
                "signed_contrast_error": recommendation_metrics.get("signed_contrast_error"),
            }
        )

    tables_dir = run_dir / "tables"
    per_state_path = write_per_state_utility_table(
        out_path=tables_dir / "utility_per_state_raw.csv",
        posters=raw_mse_out.posters,
        U_true=raw_mse_out.U_true,
        U_post_mean=raw_mse_out.U_post_mean,
        U_post_sd=raw_mse_out.U_post_sd,
        U_low=np.asarray(low),
        U_high=np.asarray(high),
    )

    per_state_norm_path = write_per_state_utility_table(
        out_path=tables_dir / "utility_per_state_posterior_draw_minmax.csv",
        posters=raw_mse_out.posters,
        U_true=raw_mse_out.U_true,
        U_post_mean=mean_norm,
        U_post_sd=np.asarray(U_norm_summary["sd"], dtype=float),
        U_low=low_norm,
        U_high=high_norm,
    )

    if state_space == "2d":
        contrast_profile_raw_path = write_contrast_profile_table(
            out_path=tables_dir / "contrast_profile_raw.csv",
            U_true=raw_mse_out.U_true,
            U_post_mean=raw_mse_out.U_post_mean,
            U_post_sd=raw_mse_out.U_post_sd,
            U_low=np.asarray(low),
            U_high=np.asarray(high),
            grid_k=int(grid_k),
        )
        contrast_profile_minmax_path = write_contrast_profile_table(
            out_path=tables_dir / "contrast_profile_posterior_draw_minmax.csv",
            U_true=raw_mse_out.U_true,
            U_post_mean=mean_norm,
            U_post_sd=np.asarray(U_norm_summary["sd"], dtype=float),
            U_low=low_norm,
            U_high=high_norm,
            grid_k=int(grid_k),
        )
    else:
        contrast_profile_raw_path = None
        contrast_profile_minmax_path = None
    
    trajectory_order_list = None if trajectory_order is None else list(trajectory_order)
    prefix_random_seed = (
        int(cfg.mcmc_main.random_seed)
        if bool(cfg.prefix.use_main_mcmc)
        else int(cfg.prefix.mcmc.random_seed)
    )
    
    summary = {
        "config": to_dict(cfg),
        "inputs": {
            "data_path": str(data_path),
            "utility_kind": utility_kind,
            "utility_metadata": utility_metadata,
            "state_space": state_space,
            "grid_k": grid_k,
            "action_order": list(action_order(state_space)),
            "preferred_contrast": (
                None if preferred_contrast is None else float(preferred_contrast)
            ),
            "model_name": str(cfg.model_name),
            "gamma_fixed": float(gamma_fixed),
            "alpha_fixed": float(getattr(cfg, "alpha_fixed", 0.0)),
            "alpha_mode": _alpha_mode(str(cfg.model_name)),
            "alpha_prior_median": float(getattr(cfg, "alpha_prior_median", 5.0)),
            "alpha_prior_log_sd": float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
            "beta_fixed": float(getattr(cfg, "beta_fixed", 0.0)),
            "hdi_prob": float(hdi_prob),
            "likelihood": likelihood,
            "prior": prior,
            "gp_eta_sd": float(cfg.gp_eta_sd),
            "gp_ls_sd": float(cfg.gp_ls_sd),
            "birl_export_mode": str(getattr(cfg, "birl_export_mode")),
            "birl_reward_scale": str(getattr(cfg, "birl_reward_scale")),
            "birl_value_iters": int(getattr(cfg, "birl_value_iters", 1000)),
        },
        "truth_utility": truth_utility_summary,
        "seed_metadata": {
            "mcmc_main_random_seed": int(cfg.mcmc_main.random_seed),
            "prefix_random_seed": int(prefix_random_seed),
            "trajectory_permutation_seed": (
                int(trajectory_permutation_seed)
                if trajectory_permutation_seed is not None
                else None
            ),
            "trajectory_order_provided": trajectory_order_list is not None,
            "trajectory_order_length": (
                len(trajectory_order_list)
                if trajectory_order_list is not None
                else None
            ),
            "trajectory_order_first_20": (
                trajectory_order_list[:20]
                if trajectory_order_list is not None
                else None
            ),
        },
        "data_summary": {
            "n_obs": (
                int(pbo_stats["n_pbo_duels"])
                if pbo_stats is not None
                else int(len(state_idx))
            ),
            "n_action_obs": int(len(state_idx)),
            "model_n_obs": (
                int(pbo_stats["n_pbo_duels"])
                if pbo_stats is not None
                else int(len(state_idx))
            ),
            "poster_min": int(cfg.poster_min),
            "poster_max": int(cfg.poster_max),
            "num_states": int(num_states),
            "unique_posters_observed": int(len(np.unique(state_idx))),
            "state_space": state_space,
            "grid_k": grid_k,
            "action_counts": _summarize_actions(action_idx, state_space=state_space),
            **(
                {
                    "pbo": {
                        "comparison_type": str(pbo_stats.get("pbo_comparison_type", _pbo_duel_label(model_name))),
                        "n_pbo_duels": int(pbo_stats["n_pbo_duels"]),
                        "n_ties_dropped": int(pbo_stats["n_ties_dropped"]),
                        "n_start_end_ties": int(pbo_stats.get("n_start_end_ties", 0)),
                        "n_transition_ties": int(pbo_stats.get("n_transition_ties", 0)),
                        "n_transition_candidates": int(pbo_stats.get("n_transition_candidates", 0)),
                        "tie_fraction": float(pbo_stats["tie_fraction"]),
                    }
                }
                if pbo_stats is not None
                else {}
            ),
        },
        "metrics": {
            "raw_mse_U_mean_vs_ground_truth": raw_mse_out.mse,
            "curve_error": curve_metrics_norm,
            "curve_error_posterior_draw_minmax": curve_metrics_norm,
            "curve_error_raw": curve_metrics_raw,
            "log_score_raw": log_score_raw,
            "ranking": ranking_metrics_norm,
            "ranking_raw": ranking_metrics_raw,
            "ranking_posterior": ranking_metrics_posterior,
            "recommendation": recommendation_metrics,
        },
        "artifacts": {
            "utility_per_state_table_csv": per_state_path,
            "utility_per_state_raw_csv": per_state_path,
            "utility_per_state_posterior_draw_minmax_csv": per_state_norm_path,
            "contrast_profile_raw_csv": contrast_profile_raw_path,
            "contrast_profile_posterior_draw_minmax_csv": contrast_profile_minmax_path,
        },
        "curves": {
            "poster": raw_mse_out.posters,
            "U_true": raw_mse_out.U_true,
            "U_post_mean": raw_mse_out.U_post_mean,
            "U_post_sd": raw_mse_out.U_post_sd,
        },
        "U_summary": {
            "mean": mean,
            "low": low,
            "high": high,
        },
        "U_summary_norm_draws": {
            "mean": mean_norm,
            "sd": np.asarray(U_norm_summary["sd"], dtype=float),
            "low": low_norm,
            "high": high_norm,
            "normalization": U_norm_summary["normalization"],
        },
    }
    if alpha_summary is not None:
        summary["alpha_summary"] = alpha_summary
    summary["soft_copeland_summary"] = {
        "mean": np.asarray(soft_copeland_summary["mean"], dtype=float),
        "sd": np.asarray(soft_copeland_summary["sd"], dtype=float),
        "low": np.asarray(soft_copeland_summary["low"], dtype=float),
        "high": np.asarray(soft_copeland_summary["high"], dtype=float),
        "preference_scale": float(soft_copeland_summary["preference_scale"]),
    }
    if main_diag is not None:
        summary["diagnostics"] = main_diag
    if main_alpha_diag is not None:
        summary["alpha_diagnostics"] = main_alpha_diag
    summary["posterior_archive"] = dict(archive_outputs)
    summary["elpd"] = elpd_outputs
    summary_path = run_dir / "summary.json"
    _save_json(summary, summary_path)

    outputs = {
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
    }
    
    outputs["utility_per_state_csv"] = per_state_path
    if contrast_profile_raw_path:
        outputs["contrast_profile_raw_csv"] = contrast_profile_raw_path
    if contrast_profile_minmax_path:
        outputs["contrast_profile_posterior_draw_minmax_csv"] = contrast_profile_minmax_path
    outputs["elpd"] = elpd_outputs
    if archive_outputs.get("enabled"):
        outputs["posterior_archive_dir"] = str(archive_dir)
        outputs["posterior_archive"] = dict(archive_outputs)

    #########
    # plots #
    #########
    if cfg.plots.enabled:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if cfg.plots.utility_curve:
            html_path = plots_dir / "posterior_utility_curve.html"
            png_path = plots_dir / "posterior_utility_curve.png"
            
            plotted_html = _plot_U_band_html(
                mean_norm,
                low_norm,
                high_norm,
                poster_min=cfg.poster_min,
                out_html=html_path,
                gamma_fixed=float(gamma_fixed),
                hdi_prob=float(hdi_prob),
                title="Posterior utility curve, posterior-draw min-max normalized:<br>"
                + _plot_title(
                    cfg,
                    gamma_fixed=float(gamma_fixed),
                    prior=prior,
                    likelihood=likelihood,
                ),
                yaxis_title="Min-max normalized utility",
                truth_y=raw_mse_out.U_true,
                truth_label=f"Ground truth U(g), {utility_kind}",
            )

            outputs["plotly_used"] = bool(plotted_html)
            outputs["U_band_html"] = str(html_path) if plotted_html else None
            outputs["U_band_minmax_html"] = str(html_path) if plotted_html else None

            raw_html_path = plots_dir / "posterior_utility_curve_raw.html"
            plotted_raw_html = _plot_U_band_html(
                mean,
                low,
                high,
                poster_min=cfg.poster_min,
                out_html=raw_html_path,
                gamma_fixed=float(gamma_fixed),
                hdi_prob=float(hdi_prob),
                title="Posterior utility curve, raw U(g):<br>"
                + _plot_title(
                    cfg,
                    gamma_fixed=float(gamma_fixed),
                    prior=prior,
                    likelihood=likelihood,
                ),
                yaxis_title="Utility U(g)",
                truth_y=raw_mse_out.U_true,
                truth_label=f"Ground truth U(g), {utility_kind}",
            )
            outputs["U_band_raw_html"] = str(raw_html_path) if plotted_raw_html else None
        
            if not plotted_html:
                plot_utility(
                    {"mean": mean, "low": low, "high": high},
                    out_path=png_path,
                    title="Posterior utility curve:<br>"
                    + _plot_title(
                        cfg,
                        gamma_fixed=float(gamma_fixed),
                        prior=prior,
                        likelihood=likelihood,
                    ),
                    show=False,
                    poster_min=int(cfg.poster_min),
                    truth_y=raw_mse_out.U_true,
                    truth_label=f"Ground truth U(g), {utility_kind}",
                )
                outputs["U_band_png"] = str(png_path)
            else:
                outputs["U_band_png"] = None

        if state_space == "2d":
            heatmap_html = plots_dir / "posterior_utility_heatmaps_2d.html"
            plot_2d_posterior_heatmaps_plotly(
                U_true=raw_mse_out.U_true,
                U_post_mean=mean_norm,
                U_post_sd=np.asarray(U_norm_summary["sd"], dtype=float),
                grid_k=int(grid_k),
                out_path_html=heatmap_html,
                title="2D posterior utility heatmaps, posterior-draw min-max normalized:<br>"
                + _plot_title(
                    cfg,
                    gamma_fixed=float(gamma_fixed),
                    prior=prior,
                    likelihood=likelihood,
                ),
                recommendation=recommendation_metrics,
                zmin=0.0,
                zmax=1.0,
            )
            outputs["posterior_utility_heatmaps_2d_html"] = str(heatmap_html)

            contrast_html = plots_dir / "contrast_profile_posterior_draw_minmax.html"
            plot_contrast_profile_plotly(
                build_contrast_profile_table(
                    U_true=raw_mse_out.U_true,
                    U_post_mean=mean_norm,
                    U_low=low_norm,
                    U_high=high_norm,
                    grid_k=int(grid_k),
                ),
                out_path_html=contrast_html,
                title="Contrast profile recovery, posterior-draw min-max normalized:<br>"
                + _plot_title(
                    cfg,
                    gamma_fixed=float(gamma_fixed),
                    prior=prior,
                    likelihood=likelihood,
                ),
                preferred_contrast=preferred_contrast,
            )
            outputs["contrast_profile_minmax_html"] = str(contrast_html)

        if cfg.plots.rank_plots:
            outputs["rank_plots_written"] = bool(_try_rank_plots(idata, plots_dir / "rank_plots"))

    if cfg.prefix.enabled:
        df = pd.DataFrame(rows)
        prefix_mcmc = cfg.mcmc_main if cfg.prefix.use_main_mcmc else cfg.prefix.mcmc

        prefix_summaries = get_or_fit_prefix_summaries(
            df=df,
            run_dir=run_dir,
            poster_min=cfg.poster_min,
            poster_max=cfg.poster_max,
            gamma_fixed=float(gamma_fixed),
            beta_fixed=float(cfg.beta_fixed),
            alpha_fixed=float(getattr(cfg, "alpha_fixed", 5.0)),
            alpha_prior_median=float(getattr(cfg, "alpha_prior_median", 5.0)),
            alpha_prior_log_sd=float(getattr(cfg, "alpha_prior_log_sd", 0.75)),
            model_name=str(getattr(cfg, "model_name", "model2_discounted")),
            likelihood=str(likelihood),
            prior=str(cfg.prior),
            gp_eta_sd=float(cfg.gp_eta_sd),
            gp_ls_sd=float(cfg.gp_ls_sd),
            draws=int(prefix_mcmc.draws),
            tune=int(prefix_mcmc.tune),
            chains=int(prefix_mcmc.chains),
            cores=int(prefix_mcmc.cores),
            target_accept=float(prefix_mcmc.target_accept),
            random_seed=int(prefix_mcmc.random_seed),
            cache=bool(cfg.prefix.cache),
            overwrite_cache=bool(cfg.prefix.overwrite_cache),
            traj_schedule=str(getattr(cfg.prefix, "train_schedule", "geom2")),
            geom_start=int(getattr(cfg.prefix, "train_geom_start", 8)),
            geom_ratio=int(getattr(cfg.prefix, "train_geom_ratio", 2)),
            k_traj_list=list(getattr(cfg.prefix, "train_k_traj_list", ())) or None,
            traj_chunk_size=int(getattr(cfg.prefix, "train_traj_chunk_size", 300)),
            hdi_prob=float(hdi_prob),
            eval_seen_df=df_test_seen,
            eval_seen_test_traj_chunk_size=int(cfg.prefix.test_traj_chunk_size),
            main_idata=idata if bool(cfg.prefix.use_main_mcmc) else None,
            reuse_main_idata_for_full_prefix=bool(cfg.prefix.use_main_mcmc),
            trajectory_order=trajectory_order,
            trajectory_permutation_seed=trajectory_permutation_seed,
            U_true_for_metrics=raw_mse_out.U_true,
            posters_for_archive=raw_mse_out.posters,
            posterior_archive_dir=archive_dir if archive_outputs.get("enabled") else None,
            posterior_archive_metadata=archive_base_metadata,
            archive_dtype=archive_dtype,
            save_u_draw_archives=save_u_draws,
            save_alpha_draw_archives=save_alpha_draws,
            save_diagnostics=save_archive_diagnostics,
            main_u_archive_path=main_u_archive_path,
            main_alpha_archive_path=main_alpha_archive_path,
            save_full_idata_archive=save_full_idata,
            main_idata_archive_path=main_idata_archive_path,
            birl_export_mode=str(getattr(cfg, "birl_export_mode")),
            birl_reward_scale=str(getattr(cfg, "birl_reward_scale")),
            birl_value_iters=int(getattr(cfg, "birl_value_iters")),
            state_space=state_space,
            grid_k=grid_k,
            preferred_contrast=preferred_contrast,
        )

        # normalized curves + metrics
        prefix_summaries_mat = materialize_prefix_summaries(
            prefix_summaries,
            utility_fn=ground_truth_utility,
            utility_kind=utility_kind,
            poster_min=int(cfg.poster_min),
            poster_max=int(cfg.poster_max),
            hdi_prob=float(hdi_prob),
            state_space=state_space,
            grid_k=grid_k,
            preferred_contrast=preferred_contrast,
        )
        
        # -----------------------------
        # Prefix ELPD artifacts (CSV)
        # -----------------------------
        prefix_elpd_rows = []
        prefix_curve_rows = []
        for s in prefix_summaries:
            if "elpd_seen_test" in s:
                ss = s["elpd_seen_test"]
                prefix_curve_rows.append(
                    {
                        "prefix_idx": int(s["k"]),
                        "k_traj": int(s["k_traj"]),
                        **{k: ss.get(k) for k in ss.keys()},
                    }
                )
            if "elpd_seen_test_per_traj" in s:
                for r in s["elpd_seen_test_per_traj"]:
                    prefix_elpd_rows.append(
                        {
                            "prefix_idx": int(s["k"]),
                            "k_traj": int(s["k_traj"]),
                            **r,
                        }
                    )

        if prefix_elpd_rows:
            out_dir = run_dir / "tables"
            out_dir.mkdir(parents=True, exist_ok=True)

            p1 = out_dir / "elpd_prefix_seen_per_traj.csv"
            pd.DataFrame(prefix_elpd_rows).to_csv(p1, index=False)
            outputs["prefix_elpd_seen_per_traj_csv"] = str(p1)

        if prefix_curve_rows:
            out_dir = run_dir / "tables"
            out_dir.mkdir(parents=True, exist_ok=True)

            p2 = out_dir / "elpd_prefix_seen_curve.csv"
            pd.DataFrame(prefix_curve_rows).to_csv(p2, index=False)
            outputs["prefix_elpd_seen_curve_csv"] = str(p2)

        prefix_path = run_dir / "prefix_summaries.materialized.json"
        _save_json({"prefix_summaries": prefix_summaries_mat}, prefix_path)
        outputs["prefix_summaries_json"] = str(prefix_path)

        metrics_csv = write_prefix_metrics_csv(prefix_summaries_mat, run_dir / "tables" / "prefix_metrics_minmax.csv")
        outputs["prefix_metrics_minmax_csv"] = metrics_csv

        alpha_prefix_csv = write_alpha_prefix_summary_csv(
            prefix_summaries_mat,
            run_dir / "tables" / "alpha_prefix_summary.csv",
        )
        if alpha_prefix_csv:
            outputs["alpha_prefix_summary_csv"] = alpha_prefix_csv

        per_state_paths = write_prefix_per_state_tables(prefix_summaries_mat, out_dir=run_dir / "tables" / "prefix_tables")
        outputs["prefix_per_state_tables_minmax_csvs"] = per_state_paths
        if state_space == "2d":
            prefix_contrast_paths = write_prefix_contrast_profile_tables(
                prefix_summaries_mat,
                out_dir=run_dir / "tables" / "prefix_contrast_profiles",
                grid_k=int(grid_k),
            )
            outputs["prefix_contrast_profile_tables_minmax_csvs"] = prefix_contrast_paths

        if cfg.plots.enabled:
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            prefix_for_plots = []
            prefix_for_raw_plots = []
            for s in prefix_summaries_mat:
                c = s["curves_minmax"]
                s_plot = dict(s)
                s_plot["mean"] = np.asarray(c["U_post_mean"], dtype=float)
                s_plot["low"] = np.asarray(c["U_hdi_low"], dtype=float)
                s_plot["high"] = np.asarray(c["U_hdi_high"], dtype=float)
                prefix_for_plots.append(s_plot)

                c_raw = s["curves"]
                s_raw_plot = dict(s)
                s_raw_plot["mean"] = np.asarray(c_raw["U_post_mean"], dtype=float)
                s_raw_plot["low"] = np.asarray(c_raw["U_hdi_low"], dtype=float)
                s_raw_plot["high"] = np.asarray(c_raw["U_hdi_high"], dtype=float)
                prefix_for_raw_plots.append(s_raw_plot)

            if cfg.plots.prefix_slider:
                out_html = plots_dir / "posterior_utility_curve_slider.html"
                plot_interactive_prefix_posterior_states_plotly(
                    prefix_for_plots,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=out_html,
                    title="Posterior of min-max normalized U(g):<br>" +
                        _plot_title(cfg, gamma_fixed=float(gamma_fixed), prior=prior, likelihood=likelihood),
                    yaxis_title="Min-max normalized utility",
                    yaxis_range=[0.0, 1.0],
                )
                outputs["prefix_slider_minmax_html"] = str(out_html)

                out_raw_html = plots_dir / "posterior_utility_curve_slider_raw.html"
                plot_interactive_prefix_posterior_states_plotly(
                    prefix_for_raw_plots,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=out_raw_html,
                    title="Posterior of raw U(g):<br>" +
                        _plot_title(cfg, gamma_fixed=float(gamma_fixed), prior=prior, likelihood=likelihood),
                    yaxis_title="Utility U(g)",
                )
                outputs["prefix_slider_raw_html"] = str(out_raw_html)

                if state_space == "2d":
                    out_2d_html = plots_dir / "posterior_utility_heatmap_slider_2d.html"
                    plot_2d_prefix_heatmap_slider_plotly(
                        prefix_for_plots,
                        grid_k=int(grid_k),
                        out_path_html=out_2d_html,
                        title="2D posterior mean utility by prefix, posterior-draw min-max normalized",
                        value_key="mean",
                        zmin=0.0,
                        zmax=1.0,
                    )
                    outputs["prefix_heatmap_slider_2d_html"] = str(out_2d_html)
                
                metrics_html = plots_dir / "rmse_interval_coverage.html"
                plot_prefix_metrics_learning_curves_plotly(
                    prefix_summaries_mat,
                    out_path_html=metrics_html,
                    title=_plot_title(
                        cfg,
                        gamma_fixed=float(gamma_fixed),
                        prior=prior,
                        likelihood=likelihood,
                        include_truth=False,
                    ),
                    use_metrics="minmax",
                )
                outputs["prefix_metrics_learning_curves_minmax_html"] = str(metrics_html)

            if cfg.plots.prefix_heatmap:
                out_html = plots_dir / "posterior_mean_heatmap.html"
                plot_prefix_mean_heatmap_states_plotly(
                    prefix_for_plots,
                    poster_min=int(cfg.poster_min),
                    poster_max=int(cfg.poster_max),
                    out_path_html=out_html,
                    title="Heatmap of min-max normalized posterior mean U(g):<br>" +
                        _plot_title(cfg, gamma_fixed=float(gamma_fixed), prior=prior, likelihood=likelihood),
                    zmin=0.0, zmax=1.0,
                )
                outputs["prefix_heatmap_minmax_html"] = str(out_html)

    if archive_outputs.get("enabled"):
        for s in locals().get("prefix_summaries", []) or []:
            if s.get("posterior_archive_u_draws_path"):
                prefix_archive_info = {}
                if s.get("posterior_archive_reuses_main") and main_archive_info is not None:
                    prefix_archive_info = {
                        "shape": main_archive_info.get("shape"),
                        "dtype": main_archive_info.get("dtype", archive_dtype),
                        "U_sha256": main_archive_info.get("U_sha256"),
                        "posters_sha256": main_archive_info.get("posters_sha256"),
                        "U_true_sha256": main_archive_info.get("U_true_sha256"),
                        "archive_file_sha256": main_archive_info.get("archive_file_sha256"),
                    }
                else:
                    prefix_archive_info = {
                        "shape": s.get("posterior_archive_shape"),
                        "dtype": s.get("posterior_archive_dtype", archive_dtype),
                        "U_sha256": s.get("posterior_archive_U_sha256"),
                        "posters_sha256": s.get("posterior_archive_posters_sha256"),
                        "U_true_sha256": s.get("posterior_archive_U_true_sha256"),
                        "archive_file_sha256": s.get("posterior_archive_file_sha256"),
                    }
                archive_entries.append(
                    {
                        "scope": "prefix",
                        "kind": "u_draws",
                        "path": str(s["posterior_archive_u_draws_path"]),
                        "k": int(s.get("k", 0)),
                        "k_traj": int(s.get("k_traj", 0)),
                        "reuses_main": bool(s.get("posterior_archive_reuses_main", False)),
                        **prefix_archive_info,
                    }
                )
            if s.get("posterior_archive_alpha_draws_path"):
                prefix_alpha_archive_info = {}
                if (
                    s.get("posterior_archive_alpha_reuses_main")
                    and main_alpha_archive_info is not None
                ):
                    prefix_alpha_archive_info = {
                        "shape": main_alpha_archive_info.get("shape"),
                        "dtype": main_alpha_archive_info.get("dtype", archive_dtype),
                        "alpha_sha256": main_alpha_archive_info.get("alpha_sha256"),
                        "log_alpha_sha256": main_alpha_archive_info.get("log_alpha_sha256"),
                        "archive_file_sha256": main_alpha_archive_info.get("archive_file_sha256"),
                    }
                else:
                    prefix_alpha_archive_info = {
                        "shape": s.get("posterior_archive_alpha_shape"),
                        "dtype": s.get("posterior_archive_alpha_dtype", archive_dtype),
                        "alpha_sha256": s.get("posterior_archive_alpha_sha256"),
                        "log_alpha_sha256": s.get("posterior_archive_log_alpha_sha256"),
                        "archive_file_sha256": s.get("posterior_archive_alpha_file_sha256"),
                    }
                archive_entries.append(
                    {
                        "scope": "prefix",
                        "kind": "alpha_draws",
                        "path": str(s["posterior_archive_alpha_draws_path"]),
                        "k": int(s.get("k", 0)),
                        "k_traj": int(s.get("k_traj", 0)),
                        "reuses_main": bool(s.get("posterior_archive_alpha_reuses_main", False)),
                        **prefix_alpha_archive_info,
                    }
                )
            if s.get("posterior_archive_idata_path"):
                archive_entries.append(
                    {
                        "scope": "prefix",
                        "kind": "full_idata",
                        "path": str(s["posterior_archive_idata_path"]),
                        "k": int(s.get("k", 0)),
                        "k_traj": int(s.get("k_traj", 0)),
                        "reuses_main": bool(s.get("posterior_archive_idata_reuses_main", False)),
                    }
                )

        prefix_diag_path = archive_dir / "diagnostics" / "prefix_diagnostics.csv"
        if prefix_diag_path.exists():
            archive_outputs["prefix_diagnostics_csv"] = str(prefix_diag_path)
        prefix_alpha_diag_path = archive_dir / "diagnostics" / "prefix_alpha_diagnostics.csv"
        if prefix_alpha_diag_path.exists():
            archive_outputs["prefix_alpha_diagnostics_csv"] = str(prefix_alpha_diag_path)

        manifest_path = write_archive_manifest(
            archive_dir=archive_dir,
            entries=archive_entries,
            metadata=archive_base_metadata,
        )
        archive_outputs["manifest_json"] = manifest_path
        summary["posterior_archive"] = dict(archive_outputs)
        _save_json(summary, summary_path)
        outputs["posterior_archive"] = dict(archive_outputs)
        outputs["posterior_archive_manifest_json"] = manifest_path

    return outputs
