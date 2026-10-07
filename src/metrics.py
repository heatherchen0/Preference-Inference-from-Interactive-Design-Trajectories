from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import csv
import numpy as np

from .state_space import (
    contrast_profile,
    normalize_state_space,
    poster_to_coord_2d,
)


@dataclass(frozen=True)
class RawMSEOutput:
    mse: float
    posters: np.ndarray
    U_true: np.ndarray
    U_post_mean: np.ndarray
    U_post_sd: np.ndarray


def compute_raw_mse_from_idata(
    idata,
    *,
    utility_fn,
    poster_min: int = 1,
    poster_max: int = 100,
    var_name: str = "U",
) -> RawMSEOutput:
    if not hasattr(idata, "posterior") or var_name not in idata.posterior:
        raise KeyError(f"Expected idata.posterior['{var_name}'] to exist.")

    posters = np.arange(int(poster_min), int(poster_max) + 1, dtype=float)  # (S,)
    U_true = np.asarray(utility_fn(posters), dtype=float)

    U_da = idata.posterior[var_name]
    U_post_mean = U_da.mean(dim=("chain", "draw")).values
    U_post_sd = U_da.std(dim=("chain", "draw")).values

    U_post_mean = np.asarray(U_post_mean, dtype=float).reshape(-1)
    U_post_sd = np.asarray(U_post_sd, dtype=float).reshape(-1)

    S = int(poster_max - poster_min + 1)
    if U_post_mean.shape[0] != S:
        raise ValueError(
            f"Posterior {var_name} has length {U_post_mean.shape[0]} but expected {S} "
            f"for poster range [{poster_min},{poster_max}]."
        )

    if U_true.shape[0] != S:
        raise ValueError(
            f"utility_fn returned length {U_true.shape[0]} but expected {S} "
            f"for poster range [{poster_min},{poster_max}]."
        )

    mse = float(np.mean((U_post_mean - U_true) ** 2))

    return RawMSEOutput(
        mse=mse,
        posters=posters.astype(int),
        U_true=U_true,
        U_post_mean=U_post_mean,
        U_post_sd=U_post_sd,
    )


def compute_curve_error_metrics(
    *,
    posters: np.ndarray,
    U_true: np.ndarray,
    U_post_mean: np.ndarray,
    U_low: np.ndarray | None = None,
    U_high: np.ndarray | None = None,
) -> dict:
    posters = np.asarray(posters)
    U_true = np.asarray(U_true, dtype=float)
    U_post_mean = np.asarray(U_post_mean, dtype=float)

    err = U_post_mean - U_true

    out = {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias_mean_error": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "U_true_mean": float(np.mean(U_true)),
        "U_post_mean_mean": float(np.mean(U_post_mean)),
    }

    if U_low is not None and U_high is not None:
        U_low = np.asarray(U_low, dtype=float)
        U_high = np.asarray(U_high, dtype=float)
        inside = (U_true >= U_low) & (U_true <= U_high)
        out["interval_coverage"] = float(np.mean(inside))
        out["interval_miss_count"] = int(np.size(inside) - int(np.sum(inside)))

    return out


def compute_gaussian_log_score_metrics(
    *,
    U_true,
    U_post_mean,
    U_post_sd,
    variance_floor: float = 1e-6,
) -> dict:
    """
    Gaussian approximation to the mean pointwise utility log score.

    The returned NLPD is the negative mean log predictive density, so lower is
    better. The variance floor is applied after squaring the posterior SD.
    """
    U_true = np.asarray(U_true, dtype=float).reshape(-1)
    U_post_mean = np.asarray(U_post_mean, dtype=float).reshape(-1)
    U_post_sd = np.asarray(U_post_sd, dtype=float).reshape(-1)

    if U_true.shape != U_post_mean.shape or U_true.shape != U_post_sd.shape:
        raise ValueError(
            "U_true, U_post_mean, and U_post_sd must have the same shape, "
            f"got {U_true.shape}, {U_post_mean.shape}, {U_post_sd.shape}"
        )

    if U_true.size == 0:
        raise ValueError("Cannot compute Gaussian log score for zero states.")

    variance_floor = float(variance_floor)
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError(f"variance_floor must be positive and finite, got {variance_floor}")

    raw_var = U_post_sd**2
    floored = raw_var < variance_floor
    var = np.maximum(raw_var, variance_floor)

    sq_error = (U_true - U_post_mean) ** 2
    pointwise_log_score = -0.5 * (np.log(2.0 * np.pi * var) + sq_error / var)
    mean_log_score = float(np.mean(pointwise_log_score))

    return {
        "gaussian_log_score": mean_log_score,
        "gaussian_nlpd": float(-mean_log_score),
        "variance_floor": variance_floor,
        "num_floored": int(np.sum(floored)),
        "fraction_floored": float(np.mean(floored)),
        "n_points": int(U_true.size),
    }


def write_per_state_utility_table(
    *,
    out_path: str | Path,
    posters: np.ndarray,
    U_true: np.ndarray,
    U_post_mean: np.ndarray,
    U_post_sd: np.ndarray | None = None,
    U_low: np.ndarray | None = None,
    U_high: np.ndarray | None = None,
) -> str:
    out_path = Path(out_path)
    posters = np.asarray(posters).astype(int)
    columns = {
        "poster": posters,
        "U_true": np.asarray(U_true, dtype=float),
        "U_post_mean": np.asarray(U_post_mean, dtype=float),
    }
    if U_post_sd is not None:
        columns["U_post_sd"] = np.asarray(U_post_sd, dtype=float)
    if U_low is not None:
        columns["U_hdi_low"] = np.asarray(U_low, dtype=float)
    if U_high is not None:
        columns["U_hdi_high"] = np.asarray(U_high, dtype=float)

    err = columns["U_post_mean"] - columns["U_true"]
    columns["error"] = err
    columns["abs_error"] = np.abs(err)
    columns["sq_error"] = err**2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix not in (".csv", ".parquet", ".pq"):
        out_path = out_path.with_suffix(".csv")

    if suffix in (".parquet", ".pq"):
        import pandas as pd

        pd.DataFrame(columns).to_parquet(out_path, index=False)
    else:
        try:
            import pandas as pd

            pd.DataFrame(columns).to_csv(out_path, index=False)
        except Exception:
            fieldnames = list(columns.keys())
            n = int(len(posters))
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(n):
                    writer.writerow({name: columns[name][i] for name in fieldnames})

    return str(out_path)

##############################################################

def compute_pairwise_accuracy(
    *,
    U_true,
    U_pred,
    eps: float = 1e-6,
) -> dict:
    U_true = np.asarray(U_true, dtype=float).reshape(-1)
    U_pred = np.asarray(U_pred, dtype=float).reshape(-1)

    if U_true.shape != U_pred.shape:
        raise ValueError(f"U_true and U_pred must have same shape, got {U_true.shape} vs {U_pred.shape}")

    n = int(U_true.size)
    if n < 2:
        return {
            "eps": float(eps),
            "n_items": n,
            "num_pairs_total": 0,
            "num_pairs_comparable": 0,
            "num_pairs_true_ties": 0,
            "num_correct_comparable": 0,
            "num_incorrect_comparable": 0,
            "pairwise_acc_comp": float("nan"),
            "pairwise_acc_strict": float("nan"),
            "num_pred_ties_total": 0,
            "num_pred_ties_among_comparable": 0,
        }

    iu, ju = np.triu_indices(n, k=1)
    dt = U_true[iu] - U_true[ju]
    dp = U_pred[iu] - U_pred[ju]

    comparable = np.abs(dt) > float(eps)
    num_total = int(dt.size)
    num_comp = int(np.sum(comparable))
    num_true_ties = int(num_total - num_comp)

    correct = comparable & (
        ((dt > eps) & (dp > eps)) |
        ((dt < -eps) & (dp < -eps))
    )
    num_correct = int(np.sum(correct))
    num_incorrect = int(num_comp - num_correct)

    pairwise_acc_comp = float(num_correct / num_comp) if num_comp > 0 else float("nan")
    pairwise_acc_strict = float(num_correct / num_total) if num_total > 0 else float("nan")

    pred_ties_total = int(np.sum(np.abs(dp) <= eps))
    pred_ties_among_comp = int(np.sum(comparable & (np.abs(dp) <= eps)))

    return {
        "eps": float(eps),
        "n_items": n,
        "num_pairs_total": num_total,
        "num_pairs_comparable": num_comp,
        "num_pairs_true_ties": num_true_ties,
        "num_correct_comparable": num_correct,
        "num_incorrect_comparable": num_incorrect,
        "pairwise_acc_comp": pairwise_acc_comp,
        "pairwise_acc_strict": pairwise_acc_strict,
        "num_pred_ties_total": pred_ties_total,
        "num_pred_ties_among_comparable": pred_ties_among_comp,
    }
    

def posterior_draw_matrix(
    idata,
    *,
    var_name: str = "U",
) -> np.ndarray:
    """
    Return posterior draws as an array of shape:

        (n_draws_total, ...var_shape)

    For U(g), this is usually:

        (n_draws_total, S)

    For scalar posterior variables, this is:

        (n_draws_total,)
    """
    if not hasattr(idata, "posterior") or var_name not in idata.posterior:
        raise KeyError(f"Expected idata.posterior['{var_name}'] to exist.")

    da = idata.posterior[var_name]
    arr = np.asarray(da.values, dtype=float)

    if arr.ndim < 2:
        raise ValueError(
            f"Expected posterior variable {var_name!r} to have dims "
            "(chain, draw, ...), got shape {arr.shape}."
        )

    n_chain, n_draw = arr.shape[:2]
    rest = arr.shape[2:]

    return arr.reshape((int(n_chain) * int(n_draw),) + rest)


def compute_posterior_pairwise_accuracy(
    *,
    U_true,
    U_draws,
    eps_true: float = 1e-6,
    eps_pred: float = 0.0,
    batch_size: int = 256,
) -> dict:
    """
    Posterior expected pairwise ranking accuracy.

    The legacy pairwise metric compares U_true against one point estimate
    U_pred. This metric keeps the posterior draw dimension and averages the
    draw-level correctness indicators over true-comparable pairs.
    """
    U_true = np.asarray(U_true, dtype=float).reshape(-1)
    draws = np.asarray(U_draws, dtype=float)

    if draws.ndim < 2:
        raise ValueError(f"U_draws must have shape (n_draws, n_states), got {draws.shape}")
    if draws.ndim > 2:
        draws = draws.reshape((draws.shape[0], -1))

    n_draws = int(draws.shape[0])
    n = int(U_true.size)
    if draws.shape[1] != n:
        raise ValueError(
            f"U_draws has {draws.shape[1]} states but U_true has {n} states."
        )

    eps_true = float(eps_true)
    eps_pred = float(eps_pred)

    if n < 2:
        return {
            "eps_true": eps_true,
            "eps_pred": eps_pred,
            "n_items": n,
            "n_draws": n_draws,
            "num_pairs_total": 0,
            "num_pairs_comparable": 0,
            "num_pairs_true_ties": 0,
            "posterior_pairwise_acc_comp": float("nan"),
            "posterior_pairwise_acc_strict": float("nan"),
            "posterior_pairwise_expected_correct_comparable": 0.0,
            "posterior_pairwise_expected_incorrect_comparable": 0.0,
            "posterior_pairwise_acc_draw_sd": float("nan"),
        }

    iu, ju = np.triu_indices(n, k=1)
    dt = U_true[iu] - U_true[ju]
    comparable = np.abs(dt) > eps_true
    num_total = int(dt.size)
    num_comp = int(np.sum(comparable))
    num_true_ties = int(num_total - num_comp)

    if n_draws <= 0:
        return {
            "eps_true": eps_true,
            "eps_pred": eps_pred,
            "n_items": n,
            "n_draws": n_draws,
            "num_pairs_total": num_total,
            "num_pairs_comparable": num_comp,
            "num_pairs_true_ties": num_true_ties,
            "posterior_pairwise_acc_comp": float("nan"),
            "posterior_pairwise_acc_strict": float("nan"),
            "posterior_pairwise_expected_correct_comparable": float("nan"),
            "posterior_pairwise_expected_incorrect_comparable": float("nan"),
            "posterior_pairwise_acc_draw_sd": float("nan"),
        }

    if num_comp == 0:
        return {
            "eps_true": eps_true,
            "eps_pred": eps_pred,
            "n_items": n,
            "n_draws": n_draws,
            "num_pairs_total": num_total,
            "num_pairs_comparable": num_comp,
            "num_pairs_true_ties": num_true_ties,
            "posterior_pairwise_acc_comp": float("nan"),
            "posterior_pairwise_acc_strict": float("nan"),
            "posterior_pairwise_expected_correct_comparable": 0.0,
            "posterior_pairwise_expected_incorrect_comparable": 0.0,
            "posterior_pairwise_acc_draw_sd": float("nan"),
        }

    comp_idx = np.flatnonzero(comparable)
    i_comp = iu[comp_idx]
    j_comp = ju[comp_idx]
    dt_comp = dt[comp_idx]

    batch_size = max(1, int(batch_size))
    total_correct = 0
    draw_acc_sum = 0.0
    draw_acc_sumsq = 0.0

    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        b = draws[start:stop]
        dp = b[:, i_comp] - b[:, j_comp]
        correct = (
            ((dt_comp > eps_true)[None, :] & (dp > eps_pred))
            |
            ((dt_comp < -eps_true)[None, :] & (dp < -eps_pred))
        )
        correct_counts = np.sum(correct, axis=1, dtype=np.int64)
        total_correct += int(np.sum(correct_counts, dtype=np.int64))
        draw_acc = correct_counts.astype(float) / float(num_comp)
        draw_acc_sum += float(np.sum(draw_acc))
        draw_acc_sumsq += float(np.sum(draw_acc**2))

    denom_comp = float(n_draws * num_comp)
    denom_total = float(n_draws * num_total)
    pairwise_acc_comp = float(total_correct / denom_comp)
    pairwise_acc_strict = float(total_correct / denom_total) if num_total > 0 else float("nan")
    expected_correct = float(total_correct / float(n_draws))
    expected_incorrect = float(num_comp - expected_correct)

    mean_draw_acc = draw_acc_sum / float(n_draws)
    var_draw_acc = max(0.0, draw_acc_sumsq / float(n_draws) - mean_draw_acc**2)

    return {
        "eps_true": eps_true,
        "eps_pred": eps_pred,
        "n_items": n,
        "n_draws": n_draws,
        "num_pairs_total": num_total,
        "num_pairs_comparable": num_comp,
        "num_pairs_true_ties": num_true_ties,
        "posterior_pairwise_acc_comp": pairwise_acc_comp,
        "posterior_pairwise_acc_strict": pairwise_acc_strict,
        "posterior_pairwise_expected_correct_comparable": expected_correct,
        "posterior_pairwise_expected_incorrect_comparable": expected_incorrect,
        "posterior_pairwise_acc_draw_sd": float(np.sqrt(var_draw_acc)),
    }


def compute_posterior_pairwise_accuracy_from_idata(
    idata,
    *,
    U_true,
    var_name: str = "U",
    eps_true: float = 1e-6,
    eps_pred: float = 0.0,
    batch_size: int = 256,
) -> dict:
    draws = posterior_draw_matrix(idata, var_name=var_name)
    if draws.ndim != 2:
        draws = draws.reshape((draws.shape[0], -1))
    return compute_posterior_pairwise_accuracy(
        U_true=U_true,
        U_draws=draws,
        eps_true=float(eps_true),
        eps_pred=float(eps_pred),
        batch_size=int(batch_size),
    )


def normalize_draws_minmax(
    draws,
    *,
    eps: float = 1e-12,
    constant_value: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Per-draw min-max normalization.

    For each posterior draw m:

        draw_norm[m, :] =
            (draw[m, :] - min_g draw[m, g]) /
            (max_g draw[m, g] - min_g draw[m, g])

    Nearly constant draws are set to constant_value.
    """
    draws = np.asarray(draws, dtype=float)

    if draws.ndim < 2:
        raise ValueError(f"draws must have shape (n_draws, ...), got {draws.shape}")

    original_shape = draws.shape
    n_draws = original_shape[0]

    flat = draws.reshape(n_draws, -1)

    lo = np.min(flat, axis=1, keepdims=True)
    hi = np.max(flat, axis=1, keepdims=True)
    denom = hi - lo

    bad = denom < float(eps)

    norm_flat = (flat - lo) / np.where(bad, 1.0, denom)

    if np.any(bad):
        norm_flat[bad[:, 0], :] = float(constant_value)

    norm = norm_flat.reshape(original_shape)

    info = {
        "method": "posterior_draw_minmax",
        "eps": float(eps),
        "constant_value_for_degenerate_draws": float(constant_value),
        "num_draws": int(n_draws),
        "num_degenerate_draws": int(np.sum(bad)),
        "fraction_degenerate_draws": float(np.mean(bad)),
    }

    return norm, info


def summarize_draws_hdi(
    draws,
    *,
    hdi_prob: float = 0.94,
) -> dict:
    """
    Summarize posterior draws with mean, sd, and HDI.
    """
    draws = np.asarray(draws, dtype=float)

    mean = np.mean(draws, axis=0)
    sd = np.std(draws, axis=0)

    try:
        import arviz as az

        hdi = az.hdi(draws, hdi_prob=float(hdi_prob))
        hdi = np.asarray(hdi, dtype=float)

        if hdi.shape[-1] != 2:
            raise ValueError(f"Unexpected HDI shape {hdi.shape}")

        low = hdi[..., 0]
        high = hdi[..., 1]

    except Exception:
        # Conservative fallback: equal-tailed interval.
        alpha = 1.0 - float(hdi_prob)
        low = np.quantile(draws, alpha / 2.0, axis=0)
        high = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)

    return {
        "mean": np.asarray(mean, dtype=float),
        "sd": np.asarray(sd, dtype=float),
        "low": np.asarray(low, dtype=float),
        "high": np.asarray(high, dtype=float),
        "hdi_prob": float(hdi_prob),
    }


def summarize_posterior_var(
    idata,
    *,
    var_name: str = "U",
    hdi_prob: float = 0.94,
    normalize_draws: bool = False,
    eps: float = 1e-12,
    constant_value: float = 0.5,
) -> dict:
    """
    Summarize posterior variable either raw or per-draw min-max normalized.

    For final utility-shape evaluation, call with normalize_draws=True.
    """
    draws = posterior_draw_matrix(idata, var_name=var_name)

    normalization = {
        "method": "none",
        "normalized": False,
    }

    if normalize_draws:
        draws, norm_info = normalize_draws_minmax(
            draws,
            eps=float(eps),
            constant_value=float(constant_value),
        )
        normalization = {
            **norm_info,
            "normalized": True,
        }

    out = summarize_draws_hdi(draws, hdi_prob=float(hdi_prob))
    out["normalization"] = normalization
    out["var_name"] = str(var_name)

    return out


def compute_recommendation_metrics(
    *,
    posters,
    U_true,
    recommendation_score,
    score_name: str = "posterior_mean_U",
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
) -> dict:
    posters = np.asarray(posters).astype(int).reshape(-1)
    U_true = np.asarray(U_true, dtype=float).reshape(-1)
    score = np.asarray(recommendation_score, dtype=float).reshape(-1)

    if posters.shape[0] != U_true.shape[0] or posters.shape[0] != score.shape[0]:
        raise ValueError(
            "posters, U_true, and recommendation_score must have the same length, "
            f"got {posters.shape[0]}, {U_true.shape[0]}, {score.shape[0]}"
        )

    if posters.shape[0] == 0:
        raise ValueError("Cannot compute recommendation metrics with zero states.")

    if not np.any(np.isfinite(score)):
        raise ValueError("recommendation_score has no finite values.")

    g_hat_idx = int(np.nanargmax(score))
    g_star_idx = int(np.nanargmax(U_true))

    U_true_at_g_hat = float(U_true[g_hat_idx])
    U_true_star = float(U_true[g_star_idx])

    out = {
        "score_name": str(score_name),
        "g_hat_idx": int(g_hat_idx),
        "g_hat": int(posters[g_hat_idx]),
        "score_at_g_hat": float(score[g_hat_idx]),
        "U_true_at_g_hat": U_true_at_g_hat,
        "g_star_idx": int(g_star_idx),
        "g_star": int(posters[g_star_idx]),
        "U_true_star": U_true_star,
        "simple_regret": float(U_true_star - U_true_at_g_hat),
    }

    if normalize_state_space(state_space) == "2d":
        if grid_k is None:
            raise ValueError("grid_k is required for 2D recommendation metrics.")
        h_hat, b_hat = poster_to_coord_2d(int(posters[g_hat_idx]), grid_k=int(grid_k))
        h_star, b_star = poster_to_coord_2d(int(posters[g_star_idx]), grid_k=int(grid_k))
        c_hat = abs(int(h_hat) - int(b_hat))
        c_star_from_optimum = abs(int(h_star) - int(b_star))
        c_star = (
            int(round(float(preferred_contrast)))
            if preferred_contrast is not None
            else int(c_star_from_optimum)
        )
        out.update(
            {
                "h_hat": int(h_hat),
                "b_hat": int(b_hat),
                "c_hat": int(c_hat),
                "h_star": int(h_star),
                "b_star": int(b_star),
                "c_star": int(c_star),
                "c_star_from_optimum": int(c_star_from_optimum),
                "contrast_error": float(abs(int(c_hat) - int(c_star))),
                "signed_contrast_error": float(int(c_hat) - int(c_star)),
            }
        )

    return out


def compute_contrast_profile_metrics(
    *,
    U_true,
    U_pred,
    grid_k: int,
) -> dict:
    true_profile = contrast_profile(U_true, grid_k=int(grid_k))
    pred_profile = contrast_profile(U_pred, grid_k=int(grid_k))
    err = pred_profile - true_profile
    return {
        "contrast_profile_rmse": float(np.sqrt(np.mean(err**2))),
        "contrast_profile_mae": float(np.mean(np.abs(err))),
        "contrast_profile_max_abs_error": float(np.max(np.abs(err))),
        "contrast_profile_num_levels": int(true_profile.shape[0]),
    }


def build_contrast_profile_table(
    *,
    U_true,
    U_post_mean,
    grid_k: int,
    U_post_sd=None,
    U_low=None,
    U_high=None,
) -> list[dict]:
    true_profile = contrast_profile(U_true, grid_k=int(grid_k))
    mean_profile = contrast_profile(U_post_mean, grid_k=int(grid_k))

    sd_profile = None if U_post_sd is None else contrast_profile(U_post_sd, grid_k=int(grid_k))
    low_profile = None if U_low is None else contrast_profile(U_low, grid_k=int(grid_k))
    high_profile = None if U_high is None else contrast_profile(U_high, grid_k=int(grid_k))

    rows = []
    for c in range(int(grid_k)):
        row = {
            "contrast": int(c),
            "U_true_profile": float(true_profile[c]),
            "U_post_mean_profile": float(mean_profile[c]),
            "error": float(mean_profile[c] - true_profile[c]),
            "abs_error": float(abs(mean_profile[c] - true_profile[c])),
        }
        if sd_profile is not None:
            row["U_post_sd_profile"] = float(sd_profile[c])
        if low_profile is not None:
            row["U_hdi_low_profile"] = float(low_profile[c])
        if high_profile is not None:
            row["U_hdi_high_profile"] = float(high_profile[c])
        rows.append(row)
    return rows


def write_contrast_profile_table(
    *,
    out_path: str | Path,
    U_true,
    U_post_mean,
    grid_k: int,
    U_post_sd=None,
    U_low=None,
    U_high=None,
) -> str:
    out_path = Path(out_path)
    rows = build_contrast_profile_table(
        U_true=U_true,
        U_post_mean=U_post_mean,
        grid_k=int(grid_k),
        U_post_sd=U_post_sd,
        U_low=U_low,
        U_high=U_high,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_path, index=False)
    except Exception:
        fieldnames = list(rows[0].keys()) if rows else []
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
    return str(out_path)


def _sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def soft_copeland_draws(
    draws,
    *,
    preference_scale: float = 1.0,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute soft-Copeland scores for utility draws.

    Input shape is (n_draws, n_states). Output has the same shape, with
    C_draw[g] = mean_h sigmoid(scale * (U_draw[g] - U_draw[h])).
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError(f"draws must have shape (n_draws, n_states), got {draws.shape}")

    n_draws, n_states = draws.shape
    out = np.empty((int(n_draws), int(n_states)), dtype=float)

    batch_size = max(1, int(batch_size))
    scale = float(preference_scale)

    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        b = draws[start:stop]
        logits = scale * (b[:, :, None] - b[:, None, :])
        out[start:stop] = np.mean(_sigmoid_np(logits), axis=2)

    return out


def summarize_soft_copeland_from_idata(
    idata,
    *,
    var_name: str = "U",
    hdi_prob: float = 0.94,
    preference_scale: float = 1.0,
    batch_size: int = 256,
) -> dict:
    draws = posterior_draw_matrix(idata, var_name=var_name)
    if draws.ndim != 2:
        draws = draws.reshape((draws.shape[0], -1))

    c_draws = soft_copeland_draws(
        draws,
        preference_scale=float(preference_scale),
        batch_size=int(batch_size),
    )
    out = summarize_draws_hdi(c_draws, hdi_prob=float(hdi_prob))
    out["var_name"] = str(var_name)
    out["score_name"] = "soft_copeland"
    out["preference_scale"] = float(preference_scale)
    return out
