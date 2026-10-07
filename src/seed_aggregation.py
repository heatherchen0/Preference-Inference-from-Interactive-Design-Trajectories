from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.plot_style import method_color, method_display_label, method_sort_key


_METRIC_LABELS = {
    "rmse": "RMSE",
    "interval_coverage": "Interval Coverage",
    "gaussian_nlpd_raw": "Gaussian NLPD",
    "posterior_pairwise_acc_comp": "Posterior Pairwise Accuracy",
    "pairwise_acc_comp": "Pairwise Accuracy",
    "simple_regret": "Simple Regret",
    "alpha_post_mean": "Alpha Posterior Mean",
    "alpha_post_sd": "Alpha Posterior SD",
    "contrast_error": "Contrast Error",
    "contrast_profile_rmse": "Contrast Profile RMSE",
}


def _metric_label(name: str) -> str:
    return _METRIC_LABELS.get(str(name), str(name))


def _color_to_rgba(color: str, alpha: float) -> str:
    c = str(color).strip()
    a = float(alpha)
    if c.startswith("rgb(") and c.endswith(")"):
        parts = c[c.find("(") + 1 : c.rfind(")")].split(",")
        r, g, b = [int(float(x.strip())) for x in parts[:3]]
        return f"rgba({r},{g},{b},{a})"
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{a})"
    return c


def _axis_range_from_values(values, *, lower_floor: float | None = 0.0):
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    pad = max(1.0, abs(hi) * 0.02) if lo == hi else max(1.0, (hi - lo) * 0.015)
    lower = lo - pad
    if lower_floor is not None:
        lower = max(float(lower_floor), lower)
    return [lower, hi + pad]


def _bounded_unit_range_with_headroom():
    return [0.0, 1.035]


DEFAULT_PREFIX_METRICS = (
    "mse",
    "rmse",
    "mae",
    "interval_coverage",
    "interval_miss_count",
    "max_abs_error",
    "gaussian_nlpd_raw",
    "gaussian_log_score_raw",
    "gaussian_nlpd_var_floor",
    "gaussian_nlpd_num_floored",
    "pairwise_acc_comp",
    "pairwise_num_pairs_comparable",
    "pairwise_num_correct_comparable",
    "posterior_pairwise_acc_comp",
    "posterior_pairwise_acc_strict",
    "posterior_pairwise_num_pairs_comparable",
    "posterior_pairwise_expected_correct_comparable",
    "posterior_pairwise_num_draws",
    "posterior_pairwise_acc_draw_sd",
    "g_hat",
    "U_true_at_g_hat",
    "simple_regret",
    "g_star",
    "U_true_star",
    "h_hat",
    "b_hat",
    "c_hat",
    "h_star",
    "b_star",
    "c_star",
    "contrast_error",
    "signed_contrast_error",
    "contrast_profile_rmse",
    "contrast_profile_mae",
    "contrast_profile_max_abs_error",
    "alpha_post_mean",
    "alpha_post_median",
    "alpha_post_sd",
    "alpha_post_q05",
    "alpha_post_q95",
    "alpha_post_low",
    "alpha_post_high",
    "alpha_diag_rhat",
    "alpha_diag_ess_bulk",
    "alpha_diag_ess_tail",
)


def _is_number_like(x) -> bool:
    try:
        if x is None:
            return False
        float(x)
        return True
    except Exception:
        return False


def build_prefix_metrics_long_df(
    entries: list[dict],
    *,
    use_metrics: str = "minmax",
) -> pd.DataFrame:
    """
    Flatten materialized prefix summaries across seeds.

    Each entry should have:
        {
            "variant_label": str,
            "seed": int,
            "mcmc_seed": int,
            "trajectory_permutation_seed": int,
            "prefix_summaries": list[dict],
        }
    """
    rows: list[dict] = []

    for e in entries:
        variant_label = str(e["variant_label"])
        seed = int(e["seed"])
        mcmc_seed = int(e["mcmc_seed"])
        trajectory_permutation_seed = int(e["trajectory_permutation_seed"])
        prefix_summaries = e.get("prefix_summaries", []) or []

        for s in prefix_summaries:
            metrics = (
                s.get("metrics", {})
                .get(str(use_metrics), {})
            )
            raw_metrics = (
                s.get("metrics", {})
                .get("raw", {})
            )

            row = {
                "variant_label": variant_label,
                "seed": seed,
                "mcmc_seed": mcmc_seed,
                "trajectory_permutation_seed": trajectory_permutation_seed,
                "prefix_idx": int(s.get("k", s.get("prefix_idx", -1))),
                "k_traj": int(s.get("k_traj", -1)),
                "prev_k_traj": int(s.get("prev_k_traj", 0)),
                "n_obs": int(s.get("n_obs", 0)),
                "n_action_obs": int(s.get("n_action_obs", s.get("n_obs", 0))),
                "n_pbo_duels": int(s.get("n_pbo_duels", 0)),
                "n_ties_dropped": int(s.get("n_ties_dropped", 0)),
                "n_traj": int(s.get("n_traj", s.get("k_traj", 0))),
                "fit_source": str(s.get("fit_source", "")),
                "reused_main_idata": bool(s.get("reused_main_idata", False)),
            }

            alpha_summary = s.get("alpha_summary", {}) or {}
            alpha_diag = s.get("alpha_diagnostics", {}) or {}
            row.update(
                {
                    "alpha_post_mean": float(alpha_summary.get("mean", np.nan)),
                    "alpha_post_median": float(alpha_summary.get("median", np.nan)),
                    "alpha_post_sd": float(alpha_summary.get("sd", np.nan)),
                    "alpha_post_q05": float(alpha_summary.get("q05", np.nan)),
                    "alpha_post_q95": float(alpha_summary.get("q95", np.nan)),
                    "alpha_post_low": float(alpha_summary.get("low", np.nan)),
                    "alpha_post_high": float(alpha_summary.get("high", np.nan)),
                    "alpha_diag_rhat": float(alpha_diag.get("rhat_max", np.nan)),
                    "alpha_diag_ess_bulk": float(alpha_diag.get("ess_bulk_min", np.nan)),
                    "alpha_diag_ess_tail": float(alpha_diag.get("ess_tail_min", np.nan)),
                }
            )

            for metric_name in DEFAULT_PREFIX_METRICS:
                if metric_name in row and _is_number_like(row[metric_name]):
                    continue
                val = metrics.get(metric_name, None)
                if not _is_number_like(val):
                    val = raw_metrics.get(metric_name, np.nan)
                row[metric_name] = float(val) if _is_number_like(val) else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_prefix_metrics_df(
    long_df: pd.DataFrame,
    *,
    metric_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate prefix metrics across seeds.

    Output columns look like:
        rmse_mean, rmse_std, rmse_sem, rmse_count
    """
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["variant_label", "prefix_idx", "k_traj"]

    if metric_cols is None:
        metric_cols = [
            c for c in DEFAULT_PREFIX_METRICS
            if c in long_df.columns
        ]

    rows: list[dict] = []

    grouped = long_df.groupby(group_cols, dropna=False, sort=True)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: key[i] for i, col in enumerate(group_cols)}

        # Useful non-metric summaries.
        if "n_obs" in g.columns:
            vals = pd.to_numeric(g["n_obs"], errors="coerce").dropna()
            row["n_obs_mean"] = float(vals.mean()) if len(vals) else np.nan

        if "n_traj" in g.columns:
            vals = pd.to_numeric(g["n_traj"], errors="coerce").dropna()
            row["n_traj_mean"] = float(vals.mean()) if len(vals) else np.nan

        if "n_pbo_duels" in g.columns:
            vals = pd.to_numeric(g["n_pbo_duels"], errors="coerce").dropna()
            row["n_pbo_duels_mean"] = float(vals.mean()) if len(vals) else np.nan

        if "n_ties_dropped" in g.columns:
            vals = pd.to_numeric(g["n_ties_dropped"], errors="coerce").dropna()
            row["n_ties_dropped_mean"] = float(vals.mean()) if len(vals) else np.nan

        seed_vals = pd.to_numeric(g["seed"], errors="coerce").dropna() if "seed" in g.columns else pd.Series(dtype=float)
        row["n_seeds"] = int(seed_vals.nunique()) if len(seed_vals) else int(len(g))

        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            n = int(len(vals))

            if n == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_sem"] = np.nan
                row[f"{metric}_count"] = 0
            else:
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if n >= 2 else 0.0
                sem = float(std / math.sqrt(n)) if n >= 2 else 0.0

                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_sem"] = sem
                row[f"{metric}_count"] = n

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_prefix_summaries_for_existing_plot(
    agg_df: pd.DataFrame,
    *,
    use_metrics: str = "minmax",
) -> dict[str, list[dict]]:
    """
    Convert aggregate metric table back into the structure expected by
    plot_prefix_metrics_learning_curves_multi_plotly.

    This plots the across-seed metric means.
    """
    if agg_df is None or len(agg_df) == 0:
        return {}

    out: dict[str, list[dict]] = {}

    for label in sorted(agg_df["variant_label"].dropna().unique()):
        sub = agg_df[agg_df["variant_label"] == label].copy()
        sub = sub.sort_values(["prefix_idx", "k_traj"], kind="stable")

        summaries = []
        for _, r in sub.iterrows():
            mm = {}
            for metric in DEFAULT_PREFIX_METRICS:
                c = f"{metric}_mean"
                if c in r:
                    mm[metric] = float(r[c])

            summaries.append(
                {
                    "k": int(r["prefix_idx"]),
                    "k_traj": int(r["k_traj"]),
                    "n_obs": int(round(float(r.get("n_obs_mean", 0.0)))) if not pd.isna(r.get("n_obs_mean", np.nan)) else 0,
                    "n_pbo_duels": int(round(float(r.get("n_pbo_duels_mean", 0.0)))) if not pd.isna(r.get("n_pbo_duels_mean", np.nan)) else 0,
                    "n_ties_dropped": int(round(float(r.get("n_ties_dropped_mean", 0.0)))) if not pd.isna(r.get("n_ties_dropped_mean", np.nan)) else 0,
                    "n_traj": int(round(float(r.get("n_traj_mean", r["k_traj"])))) if not pd.isna(r.get("n_traj_mean", np.nan)) else int(r["k_traj"]),
                    "metrics": {
                        use_metrics: mm,
                    },
                    "seed_aggregate": {
                        "n_seeds": int(r.get("n_seeds", 0)),
                    },
                }
            )

        out[str(label)] = summaries

    return out


def plot_aggregate_prefix_metrics_plotly(
    agg_df: pd.DataFrame,
    *,
    out_path_html: str | Path,
    title: str = "Prefix metrics averaged across seeds",
    metrics: tuple[str, ...] = ("rmse", "interval_coverage", "posterior_pairwise_acc_comp"),
    band: str = "sem",
) -> bool:
    """
    Plot metric means across seeds, with optional +/- SEM or +/- STD bands.

    band:
        "sem", "std", or "none"
    """
    if agg_df is None or len(agg_df) == 0:
        return False

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    available_metrics = [
        m for m in metrics
        if f"{m}_mean" in agg_df.columns
    ]
    if not available_metrics:
        return False

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(agg_df["variant_label"].dropna().unique(), key=method_sort_key)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig = make_subplots(
        rows=len(available_metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[_metric_label(m) for m in available_metrics],
    )

    use_band = str(band).lower() in ("sem", "std")
    x_range = _axis_range_from_values(pd.to_numeric(agg_df["k_traj"], errors="coerce").to_numpy())

    for row_idx, metric in enumerate(available_metrics, start=1):
        for label_idx, label in enumerate(labels):
            display_label = method_display_label(label)
            sub = agg_df[agg_df["variant_label"] == label].copy()
            sub = sub.sort_values(["k_traj", "prefix_idx"], kind="stable")

            x = sub["k_traj"].astype(float).to_numpy()
            y = sub[f"{metric}_mean"].astype(float).to_numpy()

            color = method_color(label, colors[label_idx % len(colors)])
            showlegend = row_idx == 1

            if use_band:
                band_col = f"{metric}_{str(band).lower()}"
                if band_col in sub.columns:
                    err = sub[band_col].astype(float).to_numpy()
                    if np.any(np.isfinite(err)):
                        upper = y + err
                        lower = y - err
                        bound_color = _color_to_rgba(color, 0.55)

                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=upper,
                                mode="lines+markers",
                                line=dict(width=1, color=bound_color, dash="dot"),
                                marker=dict(size=3, color=bound_color),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=display_label,
                            ),
                            row=row_idx,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=lower,
                                mode="lines+markers",
                                line=dict(width=1, color=bound_color, dash="dot"),
                                marker=dict(size=3, color=bound_color),
                                fill="tonexty",
                                fillcolor=_color_to_rgba(color, 0.14),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=display_label,
                            ),
                            row=row_idx,
                            col=1,
                        )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=showlegend,
                    line=dict(color=color, width=2),
                    marker=dict(color=color),
                ),
                row=row_idx,
                col=1,
            )

        yaxis_kwargs = {"title_text": _metric_label(metric)}
        if metric in {"interval_coverage", "posterior_pairwise_acc_comp", "pairwise_acc_comp"}:
            yaxis_kwargs["range"] = _bounded_unit_range_with_headroom()
        fig.update_yaxes(row=row_idx, col=1, **yaxis_kwargs)

    for row_idx in range(1, len(available_metrics) + 1):
        xaxis_kwargs = {
            "title_text": "Trajectory budget",
            "showticklabels": True,
        }
        if x_range is not None:
            xaxis_kwargs["range"] = x_range
        fig.update_xaxes(row=row_idx, col=1, **xaxis_kwargs)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=max(700, 300 * len(available_metrics)),
        margin=dict(l=70, r=30, t=90 if str(title).strip() else 70, b=80),
        legend=dict(
            orientation="h",
            xanchor="right",
            x=1.0,
            yanchor="bottom",
            y=1.02,
        ),
    )

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return True


def _safe_plot_filename(text: str) -> str:
    out = []
    prev_sep = False
    for ch in str(text).strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("_")
            prev_sep = True
    return "".join(out).strip("_") or "plot"


def plot_aggregate_prefix_metric_pngs(
    agg_df: pd.DataFrame,
    *,
    out_dir: str | Path,
    metrics: tuple[str, ...] = ("rmse", "interval_coverage", "posterior_pairwise_acc_comp"),
    band: str = "sem",
    filename_prefix: str = "thesis_prefix_metric",
    figsize: tuple[float, float] = (7.68, 4.48),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Write one report-oriented PNG per aggregate prefix metric.

    These are intentionally separate from the stacked Plotly dashboard so the
    report can use square-ish panels with readable legends.
    """
    if agg_df is None or len(agg_df) == 0:
        return {}

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    available_metrics = [
        m for m in metrics
        if f"{m}_mean" in agg_df.columns
    ]
    if not available_metrics:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(agg_df["variant_label"].dropna().unique(), key=method_sort_key)
    fallback_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    use_band = str(band).lower() in ("sem", "std")
    written: dict[str, str] = {}

    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=figsize)

        for label_idx, label in enumerate(labels):
            display_label = method_display_label(label)
            sub = agg_df[agg_df["variant_label"] == label].copy()
            sub = sub.sort_values(["k_traj", "prefix_idx"], kind="stable")
            if len(sub) == 0:
                continue

            x = sub["k_traj"].astype(float).to_numpy()
            y = sub[f"{metric}_mean"].astype(float).to_numpy()
            color = method_color(label, fallback_colors[label_idx % len(fallback_colors)])

            if use_band:
                band_col = f"{metric}_{str(band).lower()}"
                if band_col in sub.columns:
                    err = sub[band_col].astype(float).to_numpy()
                    if np.any(np.isfinite(err)):
                        ax.fill_between(
                            x,
                            y - err,
                            y + err,
                            color=color,
                            alpha=0.14,
                            linewidth=0,
                        )

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4.8,
                linewidth=2.0,
                color=color,
                label=display_label,
            )

        ax.set_xlabel("Trajectory budget", fontsize=13)
        ax.set_ylabel(_metric_label(metric), fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric in {"interval_coverage", "posterior_pairwise_acc_comp", "pairwise_acc_comp"}:
            ax.set_ylim(0.0, 1.035)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            frameon=False,
            fontsize=9.0,
            handlelength=2.0,
            columnspacing=1.2,
        )
        fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.31)

        out_path = out_dir / f"{filename_prefix}_{_safe_plot_filename(metric)}.png"
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        written[metric] = str(out_path)

    return written


def plot_aggregate_prefix_metric_pair_png(
    agg_df: pd.DataFrame,
    *,
    out_path_png: str | Path,
    left_metric: str = "interval_coverage",
    right_metric: str = "gaussian_nlpd_raw",
    band: str = "sem",
    figsize: tuple[float, float] = (10.5, 4.4),
    dpi: int = 300,
) -> bool:
    """
    Write a two-panel report figure with one shared legend.

    Intended for aggregate-level thesis plots where related metrics should be
    compared without using the long stacked dashboard.
    """
    if agg_df is None or len(agg_df) == 0:
        return False

    metric_names = (str(left_metric), str(right_metric))
    if any(f"{m}_mean" not in agg_df.columns for m in metric_names):
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path_png = Path(out_path_png)
    out_path_png.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(agg_df["variant_label"].dropna().unique(), key=method_sort_key)
    fallback_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    use_band = str(band).lower() in ("sem", "std")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    legend_handles = []
    legend_labels = []

    for ax, metric in zip(axes, metric_names):
        for label_idx, label in enumerate(labels):
            display_label = method_display_label(label)
            sub = agg_df[agg_df["variant_label"] == label].copy()
            sub = sub.sort_values(["k_traj", "prefix_idx"], kind="stable")
            if len(sub) == 0:
                continue

            x = sub["k_traj"].astype(float).to_numpy()
            y = sub[f"{metric}_mean"].astype(float).to_numpy()
            color = method_color(label, fallback_colors[label_idx % len(fallback_colors)])

            if use_band:
                band_col = f"{metric}_{str(band).lower()}"
                if band_col in sub.columns:
                    err = sub[band_col].astype(float).to_numpy()
                    if np.any(np.isfinite(err)):
                        ax.fill_between(
                            x,
                            y - err,
                            y + err,
                            color=color,
                            alpha=0.14,
                            linewidth=0,
                        )

            line, = ax.plot(
                x,
                y,
                marker="o",
                markersize=4.2,
                linewidth=1.9,
                color=color,
                label=display_label,
            )
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(display_label)

        ax.set_xlabel("Trajectory budget", fontsize=12)
        ax.set_ylabel(_metric_label(metric), fontsize=12)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric in {"interval_coverage", "posterior_pairwise_acc_comp", "pairwise_acc_comp"}:
            ax.set_ylim(0.0, 1.035)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        fontsize=8.8,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.30, wspace=0.26)
    fig.savefig(out_path_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return True
