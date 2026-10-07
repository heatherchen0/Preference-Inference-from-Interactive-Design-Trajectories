from __future__ import annotations
from pathlib import Path
import numpy as np
from .metrics import (
    compute_curve_error_metrics,
    write_per_state_utility_table,
    compute_pairwise_accuracy,
    compute_recommendation_metrics,
    compute_gaussian_log_score_metrics,
    compute_contrast_profile_metrics,
    write_contrast_profile_table,
)
from .state_space import normalize_state_space

def _minmax_scale_three(mean, low, high, eps: float = 1e-12):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)

    lo = float(np.min([mean.min(), low.min(), high.min()]))
    hi = float(np.max([mean.max(), low.max(), high.max()]))

    if abs(hi - lo) < eps:
        z = np.zeros_like(mean)
        return z, z.copy(), z.copy(), lo, hi, True

    mean_n = (mean - lo) / (hi - lo)
    low_n = (low - lo) / (hi - lo)
    high_n = (high - lo) / (hi - lo)
    return mean_n, low_n, high_n, lo, hi, False

def materialize_prefix_summaries(
    prefix_summaries: list[dict],
    *,
    utility_fn,
    utility_kind: str | None = None,
    poster_min: int = 1,
    poster_max: int = 100,
    hdi_prob: float = 0.94,
    pairwise_eps: float = 1e-6,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
) -> list[dict]:
    state_space = normalize_state_space(state_space)
    posters = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)
    U_true = np.asarray(utility_fn(posters), dtype=float).reshape(-1)

    out = []
    for s in prefix_summaries:
        mean = np.asarray(s["mean"], dtype=float)
        sd = np.asarray(s["sd"], dtype=float)
        low = np.asarray(s["low"], dtype=float)
        high = np.asarray(s["high"], dtype=float)

        if mean.shape[0] != U_true.shape[0] or sd.shape[0] != U_true.shape[0]:
            raise ValueError(
                f"Prefix k_traj={s.get('k_traj')} has mean len={mean.shape[0]} "
                f"and sd len={sd.shape[0]} but U_true len={U_true.shape[0]} "
                "(poster_min/max mismatch?)"
            )

        if (
            "mean_norm_draws" in s
            and "low_norm_draws" in s
            and "high_norm_draws" in s
        ):
            mean_n = np.asarray(s["mean_norm_draws"], dtype=float).reshape(-1)
            low_n = np.asarray(s["low_norm_draws"], dtype=float).reshape(-1)
            high_n = np.asarray(s["high_norm_draws"], dtype=float).reshape(-1)

            norm_info = dict(s.get("normalization_norm_draws", {}))
            lo = np.nan
            hi = np.nan
            deg = bool(norm_info.get("num_degenerate_draws", 0) > 0)
            norm_method = "posterior_draw_minmax"

        else:
            mean_n, low_n, high_n, lo, hi, deg = _minmax_scale_three(mean, low, high)
            norm_info = {
                "method": "minmax_over_{mean,low,high}",
                "normalized": True,
                "warning": "fallback_used_because_draw_normalized_summary_missing",
            }
            norm_method = "minmax_over_{mean,low,high}"

        metrics_raw = compute_curve_error_metrics(
            posters=posters.astype(int),
            U_true=U_true,
            U_post_mean=mean,
            U_low=low,
            U_high=high,
        )
        log_score_raw = compute_gaussian_log_score_metrics(
            U_true=U_true,
            U_post_mean=mean,
            U_post_sd=sd,
            variance_floor=1e-6,
        )
        metrics_raw = dict(metrics_raw)
        metrics_raw.update(
            {
                "gaussian_nlpd_raw": log_score_raw["gaussian_nlpd"],
                "gaussian_log_score_raw": log_score_raw["gaussian_log_score"],
                "gaussian_nlpd_var_floor": log_score_raw["variance_floor"],
                "gaussian_nlpd_num_floored": log_score_raw["num_floored"],
                "gaussian_nlpd_fraction_floored": log_score_raw["fraction_floored"],
                "gaussian_nlpd_n_points": log_score_raw["n_points"],
            }
        )

        metrics_minmax = compute_curve_error_metrics(
            posters=posters.astype(int),
            U_true=U_true,
            U_post_mean=mean_n,
            U_low=low_n,
            U_high=high_n,
        )

        # -------------------------
        # Pairwise ranking accuracy
        # -------------------------
        ranking = compute_pairwise_accuracy(
            U_true=U_true,
            U_pred=mean_n,
            eps=float(pairwise_eps),
        )
        posterior_ranking = {
            "posterior_pairwise_acc_comp": float(s.get("posterior_pairwise_acc_comp", np.nan)),
            "posterior_pairwise_acc_strict": float(s.get("posterior_pairwise_acc_strict", np.nan)),
            "posterior_pairwise_num_pairs_comparable": int(
                s.get("posterior_pairwise_num_pairs_comparable", 0) or 0
            ),
            "posterior_pairwise_expected_correct_comparable": float(
                s.get("posterior_pairwise_expected_correct_comparable", np.nan)
            ),
            "posterior_pairwise_num_draws": int(s.get("posterior_pairwise_num_draws", 0) or 0),
            "posterior_pairwise_eps_true": float(s.get("posterior_pairwise_eps_true", np.nan)),
            "posterior_pairwise_eps_pred": float(s.get("posterior_pairwise_eps_pred", np.nan)),
            "posterior_pairwise_acc_draw_sd": float(
                s.get("posterior_pairwise_acc_draw_sd", np.nan)
            ),
        }

        if "soft_copeland_mean" in s:
            recommendation_score = np.asarray(s["soft_copeland_mean"], dtype=float).reshape(-1)
            recommendation_score_name = "soft_copeland"
        else:
            recommendation_score = mean
            recommendation_score_name = "posterior_mean_U"

        recommendation = compute_recommendation_metrics(
            posters=posters.astype(int),
            U_true=U_true,
            recommendation_score=recommendation_score,
            score_name=recommendation_score_name,
            state_space=state_space,
            grid_k=grid_k,
            preferred_contrast=preferred_contrast,
        )

        s2 = dict(s)
        s2["hdi_prob"] = float(hdi_prob)

        if utility_kind is not None:
            s2["utility_kind"] = str(utility_kind)
    
        s2["curves"] = {
            "poster": posters.astype(int),
            "utility_kind": None if utility_kind is None else str(utility_kind),
            "U_true": U_true,
            "U_post_mean": mean,
            "U_hdi_low": low,
            "U_hdi_high": high,
        }

        s2["curves_minmax"] = {
            "poster": posters.astype(int),
            "utility_kind": None if utility_kind is None else str(utility_kind),
            "U_true": U_true,
            "U_post_mean": mean_n,
            "U_hdi_low": low_n,
            "U_hdi_high": high_n,
            "scale_min": lo,
            "scale_max": hi,
            "degenerate": bool(deg),
            "method": norm_method,
            "normalization": norm_info,
        }

        metrics_minmax = dict(metrics_minmax)
        metrics_minmax.update(
            {
                "pairwise_acc_comp": ranking["pairwise_acc_comp"],
                "pairwise_num_pairs_comparable": ranking["num_pairs_comparable"],
                "pairwise_num_correct_comparable": ranking["num_correct_comparable"],
                "pairwise_eps": ranking["eps"],
                **posterior_ranking,
                "g_hat": recommendation["g_hat"],
                "U_true_at_g_hat": recommendation["U_true_at_g_hat"],
                "simple_regret": recommendation["simple_regret"],
                "g_star": recommendation["g_star"],
                "U_true_star": recommendation["U_true_star"],
            }
        )
        if state_space == "2d":
            profile_metrics = compute_contrast_profile_metrics(
                U_true=U_true,
                U_pred=mean_n,
                grid_k=int(grid_k),
            )
            metrics_minmax.update(profile_metrics)
            metrics_minmax.update(
                {
                    "h_hat": recommendation.get("h_hat"),
                    "b_hat": recommendation.get("b_hat"),
                    "c_hat": recommendation.get("c_hat"),
                    "h_star": recommendation.get("h_star"),
                    "b_star": recommendation.get("b_star"),
                    "c_star": recommendation.get("c_star"),
                    "contrast_error": recommendation.get("contrast_error"),
                    "signed_contrast_error": recommendation.get("signed_contrast_error"),
                }
            )

        s2["metrics"] = {
            "raw": metrics_raw,
            "minmax": metrics_minmax,
            "ranking": ranking,
            "ranking_posterior": posterior_ranking,
            "recommendation": recommendation,
        }
        s2["recommendation"] = recommendation

        out.append(s2)

    return out


def write_prefix_metrics_csv(prefix_summaries_mat: list[dict], out_path: str | Path) -> str:
    import pandas as pd

    out_path = Path(out_path)
    rows = []
    for s in prefix_summaries_mat:
        mm = s["metrics"]["minmax"]
        raw = s["metrics"].get("raw", {})
        alpha_summary = s.get("alpha_summary", {}) or {}
        rows.append(
            {
                "prefix_idx": int(s["k"]),
                "k_traj": int(s["k_traj"]),
                "utility_kind": str(s.get("utility_kind", "")),
                "prev_k_traj": int(s.get("prev_k_traj", 0)),
                "n_obs": int(s["n_obs"]),
                "n_action_obs": int(s.get("n_action_obs", s.get("n_obs", 0))),
                "n_pbo_duels": int(s.get("n_pbo_duels", 0)),
                "n_ties_dropped": int(s.get("n_ties_dropped", 0)),
                "alpha_mode": str(s.get("alpha_mode", "")),
                "alpha_post_mean": float(alpha_summary.get("mean", np.nan)),
                "alpha_post_median": float(alpha_summary.get("median", np.nan)),
                "alpha_post_sd": float(alpha_summary.get("sd", np.nan)),
                "alpha_post_q05": float(alpha_summary.get("q05", np.nan)),
                "alpha_post_q95": float(alpha_summary.get("q95", np.nan)),
                "alpha_post_low": float(alpha_summary.get("low", np.nan)),
                "alpha_post_high": float(alpha_summary.get("high", np.nan)),
                "alpha_prior_median": float(s.get("alpha_prior_median", alpha_summary.get("prior_median", np.nan))),
                "alpha_prior_log_sd": float(s.get("alpha_prior_log_sd", alpha_summary.get("prior_log_sd", np.nan))),
                "new_traj_start_posters": ",".join(map(str, s.get("new_traj_start_posters", []))),
                "mse": float(mm.get("mse", np.nan)),
                "rmse": float(mm.get("rmse", np.nan)),
                "mae": float(mm.get("mae", np.nan)),
                "interval_coverage": float(mm.get("interval_coverage", np.nan)),
                "interval_miss_count": float(mm.get("interval_miss_count", np.nan)),
                "max_abs_error": float(mm.get("max_abs_error", np.nan)),
                "gaussian_nlpd_raw": float(raw.get("gaussian_nlpd_raw", np.nan)),
                "gaussian_log_score_raw": float(raw.get("gaussian_log_score_raw", np.nan)),
                "gaussian_nlpd_var_floor": float(raw.get("gaussian_nlpd_var_floor", np.nan)),
                "gaussian_nlpd_num_floored": int(raw.get("gaussian_nlpd_num_floored", 0) or 0),

                "pairwise_acc_comp": float(mm.get("pairwise_acc_comp", np.nan)),
                "pairwise_num_pairs_comparable": int(mm.get("pairwise_num_pairs_comparable", 0) or 0),
                "pairwise_num_correct_comparable": int(mm.get("pairwise_num_correct_comparable", 0) or 0),
                "pairwise_eps": float(mm.get("pairwise_eps", np.nan)),
                "posterior_pairwise_acc_comp": float(mm.get("posterior_pairwise_acc_comp", np.nan)),
                "posterior_pairwise_acc_strict": float(mm.get("posterior_pairwise_acc_strict", np.nan)),
                "posterior_pairwise_num_pairs_comparable": int(
                    mm.get("posterior_pairwise_num_pairs_comparable", 0) or 0
                ),
                "posterior_pairwise_expected_correct_comparable": float(
                    mm.get("posterior_pairwise_expected_correct_comparable", np.nan)
                ),
                "posterior_pairwise_num_draws": int(mm.get("posterior_pairwise_num_draws", 0) or 0),
                "posterior_pairwise_eps_true": float(mm.get("posterior_pairwise_eps_true", np.nan)),
                "posterior_pairwise_eps_pred": float(mm.get("posterior_pairwise_eps_pred", np.nan)),
                "posterior_pairwise_acc_draw_sd": float(
                    mm.get("posterior_pairwise_acc_draw_sd", np.nan)
                ),
                "g_hat": int(mm.get("g_hat", 0) or 0),
                "U_true_at_g_hat": float(mm.get("U_true_at_g_hat", np.nan)),
                "simple_regret": float(mm.get("simple_regret", np.nan)),
                "g_star": int(mm.get("g_star", 0) or 0),
                "U_true_star": float(mm.get("U_true_star", np.nan)),
                "h_hat": int(mm.get("h_hat", 0) or 0),
                "b_hat": int(mm.get("b_hat", 0) or 0),
                "c_hat": int(mm.get("c_hat", 0) or 0),
                "h_star": int(mm.get("h_star", 0) or 0),
                "b_star": int(mm.get("b_star", 0) or 0),
                "c_star": int(mm.get("c_star", 0) or 0),
                "contrast_error": float(mm.get("contrast_error", np.nan)),
                "signed_contrast_error": float(mm.get("signed_contrast_error", np.nan)),
                "contrast_profile_rmse": float(mm.get("contrast_profile_rmse", np.nan)),
                "contrast_profile_mae": float(mm.get("contrast_profile_mae", np.nan)),
                "contrast_profile_max_abs_error": float(
                    mm.get("contrast_profile_max_abs_error", np.nan)
                ),
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def write_alpha_prefix_summary_csv(
    prefix_summaries_mat: list[dict],
    out_path: str | Path,
) -> str | None:
    import pandas as pd

    def _float_or_nan(value) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _int_or_nan(value):
        if value is None:
            return np.nan
        try:
            return int(value)
        except Exception:
            return np.nan

    rows = []
    for s in prefix_summaries_mat:
        alpha_summary = s.get("alpha_summary", {}) or {}
        if not alpha_summary:
            continue
        diag = s.get("alpha_diagnostics", {}) or {}
        rows.append(
            {
                "prefix_idx": int(s["k"]),
                "k_traj": int(s["k_traj"]),
                "prev_k_traj": int(s.get("prev_k_traj", 0)),
                "n_obs": int(s["n_obs"]),
                "n_action_obs": int(s.get("n_action_obs", s.get("n_obs", 0))),
                "n_traj": int(s.get("n_traj", s.get("k_traj", 0))),
                "fit_source": str(s.get("fit_source", "")),
                "reused_main_idata": bool(s.get("reused_main_idata", False)),
                "alpha_mode": str(s.get("alpha_mode", "")),
                "alpha_post_mean": _float_or_nan(alpha_summary.get("mean", np.nan)),
                "alpha_post_median": _float_or_nan(alpha_summary.get("median", np.nan)),
                "alpha_post_sd": _float_or_nan(alpha_summary.get("sd", np.nan)),
                "alpha_post_q05": _float_or_nan(alpha_summary.get("q05", np.nan)),
                "alpha_post_q95": _float_or_nan(alpha_summary.get("q95", np.nan)),
                "alpha_post_low": _float_or_nan(alpha_summary.get("low", np.nan)),
                "alpha_post_high": _float_or_nan(alpha_summary.get("high", np.nan)),
                "alpha_hdi_prob": _float_or_nan(alpha_summary.get("hdi_prob", np.nan)),
                "alpha_prior": str(alpha_summary.get("prior", "")),
                "alpha_prior_median": _float_or_nan(
                    s.get("alpha_prior_median", alpha_summary.get("prior_median", np.nan))
                ),
                "alpha_prior_log_sd": _float_or_nan(
                    s.get("alpha_prior_log_sd", alpha_summary.get("prior_log_sd", np.nan))
                ),
                "alpha_diag_rhat": _float_or_nan(diag.get("rhat_max", np.nan)),
                "alpha_diag_ess_bulk": _float_or_nan(diag.get("ess_bulk_min", np.nan)),
                "alpha_diag_ess_tail": _float_or_nan(diag.get("ess_tail_min", np.nan)),
                "alpha_diag_divergence_count": _int_or_nan(diag.get("divergence_count")),
                "alpha_diag_divergence_rate": _float_or_nan(diag.get("divergence_rate", np.nan)),
                "alpha_diag_bfmi_min": _float_or_nan(diag.get("bfmi_min", np.nan)),
                "posterior_archive_alpha_draws_path": str(
                    s.get("posterior_archive_alpha_draws_path", "")
                ),
                "posterior_archive_alpha_reuses_main": bool(
                    s.get("posterior_archive_alpha_reuses_main", False)
                ),
            }
        )

    if not rows:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)

def write_prefix_per_state_tables(
    prefix_summaries_mat: list[dict],
    *,
    out_dir: str | Path,
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for s in prefix_summaries_mat:
        k_traj = int(s["k_traj"])
        c = s["curves_minmax"]
        p = write_per_state_utility_table(
            out_path=out_dir / f"prefix_{k_traj:04d}_utility_per_state_minmax.csv",
            posters=np.asarray(c["poster"], dtype=int),
            U_true=np.asarray(c["U_true"], dtype=float),
            U_post_mean=np.asarray(c["U_post_mean"], dtype=float),
            U_post_sd=None,
            U_low=np.asarray(c["U_hdi_low"], dtype=float),
            U_high=np.asarray(c["U_hdi_high"], dtype=float),
        )
        paths.append(p)
    return paths


def write_prefix_contrast_profile_tables(
    prefix_summaries_mat: list[dict],
    *,
    out_dir: str | Path,
    grid_k: int,
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for s in prefix_summaries_mat:
        k_traj = int(s["k_traj"])
        c = s["curves_minmax"]
        p = write_contrast_profile_table(
            out_path=out_dir / f"prefix_{k_traj:04d}_contrast_profile_minmax.csv",
            U_true=np.asarray(c["U_true"], dtype=float),
            U_post_mean=np.asarray(c["U_post_mean"], dtype=float),
            U_low=np.asarray(c["U_hdi_low"], dtype=float),
            U_high=np.asarray(c["U_hdi_high"], dtype=float),
            grid_k=int(grid_k),
        )
        paths.append(p)
    return paths
