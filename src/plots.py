from pathlib import Path
import numpy as np

from src.plot_style import method_color, method_display_label, method_sort_key


_METRIC_LABELS = {
    "rmse": "RMSE",
    "interval_coverage": "Interval Coverage",
    "gaussian_nlpd_raw": "Gaussian NLPD",
    "posterior_pairwise_acc_comp": "Posterior Pairwise Accuracy",
    "pairwise_acc_comp": "Pairwise Accuracy",
    "simple_regret": "Simple Regret",
    "contrast_error": "Contrast Error",
    "contrast_profile_rmse": "Contrast Profile RMSE",
}


def _metric_label(name: str) -> str:
    return _METRIC_LABELS.get(str(name), str(name))


def _axis_range_from_values(values, *, lower_floor: float | None = 0.0):
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo == hi:
        pad = max(1.0, abs(hi) * 0.02)
    else:
        pad = max(1.0, (hi - lo) * 0.015)

    lower = lo - pad
    upper = hi + pad
    if lower_floor is not None:
        lower = max(float(lower_floor), lower)
    return [lower, upper]


def _show_x_axis_on_all_rows(fig, *, n_rows: int, title_text: str, x_values):
    x_range = _axis_range_from_values(x_values)
    for row in range(1, int(n_rows) + 1):
        kwargs = {
            "title_text": str(title_text),
            "showticklabels": True,
        }
        if x_range is not None:
            kwargs["range"] = x_range
        fig.update_xaxes(row=row, col=1, **kwargs)


def _bounded_unit_range_with_headroom():
    return [0.0, 1.035]


def plot_utility(
    U_summary,
    out_path: Path,
    title="Utility",
    show=False,
    poster_min: int = 0,
    truth_y=None,
    truth_label: str = "Ground truth",
    yaxis_title: str = "Utility U(g)",
):
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mean = np.asarray(U_summary["mean"])
    low = np.asarray(U_summary["low"])
    high = np.asarray(U_summary["high"])

    plt.figure(figsize=(10, 4))
    if mean.ndim == 1 and mean.shape[0] > 3:
        x = np.arange(mean.shape[0]) + int(poster_min)
        plt.plot(x, mean, label="Posterior mean U")
        plt.fill_between(x, low, high, alpha=0.2, label="Interval")
        plt.xlabel("Poster state g")
    elif mean.ndim == 2:
        x = np.arange(mean.shape[0]) + int(poster_min)
        for a, name in enumerate(["left", "stay", "right"]):
            plt.plot(x, mean[:, a], label=f"{name} mean")
            plt.fill_between(x, low[:, a], high[:, a], alpha=0.2)
        plt.xlabel("Poster state g")
    else:
        x = np.arange(mean.shape[0]) + int(poster_min)
        for a, name in enumerate(["left", "stay", "right"]):
            plt.errorbar([a], [mean[a]], yerr=[[mean[a] - low[a]], [high[a] - mean[a]]], fmt="o", label=name)
        plt.xticks([0, 1, 2], ["left", "stay", "right"])

    if truth_y is not None and (mean.ndim == 2 or (mean.ndim == 1 and mean.shape[0] > 3)):
        truth_y = np.asarray(truth_y, dtype=float).reshape(-1)
        if truth_y.shape[0] == mean.shape[0]:
            plt.plot(x, truth_y, color="black", linewidth=2, label=str(truth_label))

    plt.title(title)
    plt.ylabel(str(yaxis_title))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()

def prefix_mean_matrix_states(prefix_summaries):
    if len(prefix_summaries) == 0:
        raise ValueError("prefix_mean_matrix_states: prefix_summaries is empty.")
    means = [np.asarray(s["mean"], dtype=float) for s in prefix_summaries]
    num_states = means[0].shape[0]
    for i, m in enumerate(means):
        if m.shape[0] != num_states:
            raise ValueError(f"inconsistent num_states at i={i}: {m.shape[0]} vs {num_states}")
    Z = np.stack(means, axis=0).T
    return Z

def plot_interactive_prefix_posterior_states_plotly(prefix_summaries, poster_min: int, poster_max: int,
                                                   out_path_html: Path,
                                                   title="Prefix posterior U(g) by k",
                                                   yaxis_title: str = "Utility U(g)",
                                                   yaxis_range=None):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    g_vals = np.arange(poster_min, poster_max + 1)
    n_prefix = len(prefix_summaries)

    fig = go.Figure()

    for i, s in enumerate(prefix_summaries):
        visible = (i == 0)
        low = np.asarray(s["low"], dtype=float)
        high = np.asarray(s["high"], dtype=float)
        mean = np.asarray(s["mean"], dtype=float)

        # low
        fig.add_trace(go.Scatter(
            x=g_vals, y=low, mode="lines", line=dict(width=0),
            showlegend=False, visible=visible, hoverinfo="skip"
        ))
        # high
        fig.add_trace(go.Scatter(
            x=g_vals, y=high, mode="lines",
            fill="tonexty", fillcolor="rgba(31,119,180,0.20)",
            line=dict(width=0), showlegend=False, visible=visible, hoverinfo="skip"
        ))
        # mean
        fig.add_trace(go.Scatter(
            x=g_vals, y=mean, mode="lines",
            line=dict(width=2, color="rgba(31,119,180,1.0)"),
            name="Posterior mean", showlegend=(i == 0), visible=visible,
            hovertemplate="g=%{x}<br>U=%{y:.3f}<extra></extra>",
        ))

    steps = []
    for i, s in enumerate(prefix_summaries):
        vis = [False] * (3 * n_prefix)
        vis[3*i:3*i+3] = [True, True, True]

        traj_ids = s.get("traj_ids", [])
        traj_label = ""
        if len(traj_ids) >= 2:
            traj_label = f"{traj_ids[0]}..{traj_ids[-1]}"
        elif len(traj_ids) == 1:
            traj_label = f"{traj_ids[0]}"

        step_label = f"k={s.get('k', i+1)}, n={s.get('n_obs', '')}, traj={traj_label}"
        steps.append(dict(
            method="update",
            label=f"k={s.get('k', i+1)}",
            args=[{"visible": vis}, {"title": f"{title}<br><sup>{step_label}</sup>"}],
        ))

    first = prefix_summaries[0]
    fig.update_layout(
        title=f"{title}<br><sup>k={first.get('k', 1)}, n={first.get('n_obs','')}</sup>",
        xaxis_title="Poster state g",
        yaxis_title=str(yaxis_title),
        template="plotly_white",
        sliders=[dict(active=0, steps=steps, x=0.05, xanchor="left", y=0.0, yanchor="top")],
        margin=dict(l=60, r=30, t=80, b=80),
        height=520,
    )
    if yaxis_range is not None:
        fig.update_yaxes(range=list(yaxis_range))

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig

def plot_prefix_mean_heatmap_states_plotly(prefix_summaries, poster_min: int, poster_max: int,
                                          out_path_html: Path,
                                          title="Prefix mean utility heatmap",
                                          colorscale="Viridis",
                                          zmin=None, zmax=None):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    Z = prefix_mean_matrix_states(prefix_summaries)  # (num_states, n_prefix)
    ks = np.arange(1, len(prefix_summaries) + 1)
    g_vals = np.arange(poster_min, poster_max + 1)

    n_obs = [s.get("n_obs", None) for s in prefix_summaries]
    traj_first_last = []
    for s in prefix_summaries:
        tids = s.get("traj_ids", [])
        if len(tids) >= 2:
            traj_first_last.append(f"{tids[0]}..{tids[-1]}")
        elif len(tids) == 1:
            traj_first_last.append(f"{tids[0]}")
        else:
            traj_first_last.append("")

    customdata = np.zeros((len(g_vals), len(ks), 3), dtype=object)
    for j, k in enumerate(ks):
        customdata[:, j, 0] = int(k)
        customdata[:, j, 1] = n_obs[j]
        customdata[:, j, 2] = traj_first_last[j]

    fig = go.Figure(go.Heatmap(
        x=ks, y=g_vals, z=Z,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title="mean U(g)"),
        customdata=customdata,
        hovertemplate=(
            "g=%{y}<br>"
            "k=%{customdata[0]}<br>"
            "mean U=%{z:.4f}<br>"
            "n=%{customdata[1]}<br>"
            "traj=%{customdata[2]}<extra></extra>"
        )
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Trajectory k",
        yaxis_title="Poster state g",
        template="plotly_white",
        margin=dict(l=70, r=30, t=70, b=60),
        height=650,
    )

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig


def plot_prefix_metrics_learning_curves_plotly(
    prefix_summaries,
    *,
    out_path_html: Path,
    title: str = "RMSE, interval coverage, Gaussian NLPD, pairwise accuracy, and simple regret",
    use_metrics: str = "minmax",   # "minmax" or "raw"
    include_gaussian_nlpd: bool = True,
    include_pairwise_acc: bool = True,
    include_recommendation_metrics: bool = True,
    include_2d_metrics: bool = True,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    def _get_metric(s, key: str):
        if key in s:
            return s.get(key)

        m = s.get("metrics", {})
        if not isinstance(m, dict):
            return None

        if use_metrics in m and isinstance(m[use_metrics], dict) and key in m[use_metrics]:
            return m[use_metrics].get(key)

        mm = m.get("minmax", {})
        if isinstance(mm, dict) and key in mm:
            return mm.get(key)
        raw = m.get("raw", {})
        if isinstance(raw, dict) and key in raw:
            return raw.get(key)
        return None

    def _as_float(x):
        try:
            if x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _get_pairwise_acc(s):
        val = _as_float(_get_metric(s, "posterior_pairwise_acc_comp"))
        if np.isnan(val):
            val = _as_float(_get_metric(s, "pairwise_acc_comp"))
        return val

    xs, ks, n_obs, n_pbo, n_ties = [], [], [], [], []
    rmse, gaussian_nlpd, cov, pairacc, regret = [], [], [], [], []
    contrast_error, contrast_profile_rmse = [], []

    for i, s in enumerate(prefix_summaries):
        ks.append(int(s.get("k", i + 1)))
        n_obs.append(int(s.get("n_obs", 0)))
        n_pbo.append(int(s.get("n_pbo_duels", 0)))
        n_ties.append(int(s.get("n_ties_dropped", 0)))

        x = s.get("k_traj", None)
        if x is None:
            x = s.get("n_traj", None)
        if x is None:
            x = s.get("k", i + 1)
        xs.append(int(x))

        rmse.append(_as_float(_get_metric(s, "rmse")))
        gaussian_nlpd.append(_as_float(_get_metric(s, "gaussian_nlpd_raw")))
        cov.append(_as_float(_get_metric(s, "interval_coverage")))
        pairacc.append(_get_pairwise_acc(s))
        regret.append(_as_float(_get_metric(s, "simple_regret")))
        contrast_error.append(_as_float(_get_metric(s, "contrast_error")))
        contrast_profile_rmse.append(_as_float(_get_metric(s, "contrast_profile_rmse")))

    xs = np.asarray(xs, dtype=int)
    rmse = np.asarray(rmse, dtype=float)
    gaussian_nlpd = np.asarray(gaussian_nlpd, dtype=float)
    cov = np.asarray(cov, dtype=float)
    pairacc = np.asarray(pairacc, dtype=float)
    regret = np.asarray(regret, dtype=float)
    contrast_error = np.asarray(contrast_error, dtype=float)
    contrast_profile_rmse = np.asarray(contrast_profile_rmse, dtype=float)
    has_2d_metrics = bool(
        include_2d_metrics
        and (
            np.any(np.isfinite(contrast_error))
            or np.any(np.isfinite(contrast_profile_rmse))
        )
    )

    custom = np.stack(
        [
            np.asarray(ks, dtype=int),
            np.asarray(n_obs, dtype=int),
            np.asarray(n_pbo, dtype=int),
            np.asarray(n_ties, dtype=int),
        ],
        axis=1,
    )

    subplot_titles = [_metric_label("rmse"), _metric_label("interval_coverage")]
    if include_gaussian_nlpd:
        subplot_titles.append(_metric_label("gaussian_nlpd_raw"))
    if include_pairwise_acc:
        subplot_titles.append(_metric_label("posterior_pairwise_acc_comp"))
    if include_recommendation_metrics:
        subplot_titles.append(_metric_label("simple_regret"))
    if has_2d_metrics:
        subplot_titles.extend([
            _metric_label("contrast_error"),
            _metric_label("contrast_profile_rmse"),
        ])
    n_rows = len(subplot_titles)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Scatter(
            x=xs, y=rmse,
            mode="lines+markers",
            name="RMSE",
            marker=dict(size=7),
            hovertemplate=(
                "k_traj=%{x}<br>"
                "rmse=%{y:.6f}<br>"
                "k=%{customdata[0]}<br>"
                "n_obs=%{customdata[1]}<br>"
                "n_pbo_duels=%{customdata[2]}<br>"
                "n_ties_dropped=%{customdata[3]}<extra></extra>"
            ),
            customdata=custom,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=xs, y=cov,
            mode="lines+markers",
            name=_metric_label("interval_coverage"),
            marker=dict(size=7),
            hovertemplate=(
                "k_traj=%{x}<br>"
                "coverage=%{y:.4f}<br>"
                "k=%{customdata[0]}<br>"
                "n_obs=%{customdata[1]}<br>"
                "n_pbo_duels=%{customdata[2]}<br>"
                "n_ties_dropped=%{customdata[3]}<extra></extra>"
            ),
            customdata=custom,
        ),
        row=2, col=1
    )

    fig.update_yaxes(title_text=_metric_label("rmse"), row=1, col=1)
    fig.update_yaxes(
        title_text=_metric_label("interval_coverage"),
        range=_bounded_unit_range_with_headroom(),
        row=2,
        col=1,
    )

    next_row = 3
    if include_gaussian_nlpd:
        fig.add_trace(
            go.Scatter(
                x=xs, y=gaussian_nlpd,
                mode="lines+markers",
                name=_metric_label("gaussian_nlpd_raw"),
                marker=dict(size=7),
                hovertemplate=(
                    "k_traj=%{x}<br>"
                    "Gaussian NLPD=%{y:.6f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=next_row, col=1
        )
        fig.update_yaxes(title_text=_metric_label("gaussian_nlpd_raw"), row=next_row, col=1)
        next_row += 1

    if include_pairwise_acc:
        fig.add_trace(
            go.Scatter(
                x=xs, y=pairacc,
                mode="lines+markers",
                name=_metric_label("posterior_pairwise_acc_comp"),
                marker=dict(size=7),
                hovertemplate=(
                    "k_traj=%{x}<br>"
                    "posterior_pairwise_acc=%{y:.4f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=next_row, col=1
        )
        fig.update_yaxes(
            title_text=_metric_label("posterior_pairwise_acc_comp"),
            range=_bounded_unit_range_with_headroom(),
            row=next_row,
            col=1,
        )
        next_row += 1

    if include_recommendation_metrics:
        fig.add_trace(
            go.Scatter(
                x=xs, y=regret,
                mode="lines+markers",
                name=_metric_label("simple_regret"),
                marker=dict(size=7),
                hovertemplate=(
                    "k_traj=%{x}<br>"
                    "simple_regret=%{y:.4f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=next_row, col=1
        )
        fig.update_yaxes(title_text=_metric_label("simple_regret"), row=next_row, col=1)
        next_row += 1

    if has_2d_metrics:
        fig.add_trace(
            go.Scatter(
                x=xs, y=contrast_error,
                mode="lines+markers",
                name=_metric_label("contrast_error"),
                marker=dict(size=7),
                hovertemplate=(
                    "k_traj=%{x}<br>"
                    "contrast_error=%{y:.4f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=next_row,
            col=1,
        )
        fig.update_yaxes(title_text=_metric_label("contrast_error"), row=next_row, col=1)
        next_row += 1

        fig.add_trace(
            go.Scatter(
                x=xs, y=contrast_profile_rmse,
                mode="lines+markers",
                name=_metric_label("contrast_profile_rmse"),
                marker=dict(size=7),
                hovertemplate=(
                    "k_traj=%{x}<br>"
                    "contrast_profile_rmse=%{y:.6f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=next_row,
            col=1,
        )
        fig.update_yaxes(title_text=_metric_label("contrast_profile_rmse"), row=next_row, col=1)

    _show_x_axis_on_all_rows(
        fig,
        n_rows=n_rows,
        title_text="Number of trajectories (cumulative)",
        x_values=xs,
    )

    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.985, xanchor="left", yanchor="top"),
        template="plotly_white",
        height=max(700, 280 * n_rows),
        margin=dict(l=70, r=30, t=135, b=70),
        legend=dict(
            orientation="h",
            xanchor="right",
            x=1.0,
            yanchor="bottom",
            y=1.02,
        ),
    )

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig

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


def _prepare_posterior_overlay_series(series):
    return sorted(
        list(series),
        key=lambda item: method_sort_key(item.get("label", ""))
        if isinstance(item, dict)
        else method_sort_key(""),
    )


def plot_posterior_U_overlay_png(
    series,
    *,
    poster_min: int,
    poster_max: int,
    out_path_png: Path,
    scale_mode: str = "raw",
    yaxis_title: str | None = None,
    yaxis_range=None,
    band_alpha: float = 0.14,
    band_source: str = "default",
    show_hdi_band: bool = True,
    truth_y=None,
    truth_label: str = "Ground truth",
    truth_color: str = "black",
    truth_dash: str = "solid",
    truth_width: float = 2.8,
    line_width: float = 2.2,
    legend_fontsize: float = 10.5,
    legend_ncol: int = 2,
    figsize: tuple[float, float] = (6.4, 5.6),
    dpi: int = 300,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path_png = Path(out_path_png)
    out_path_png.parent.mkdir(parents=True, exist_ok=True)

    g_vals = np.arange(int(poster_min), int(poster_max) + 1, dtype=int)
    mode = str(scale_mode).strip().lower()
    if mode in ("minmax", "posterior_minmax", "posterior_draw_minmax"):
        mode = "posterior_draw_minmax"
    elif mode in ("raw", "raw_u", "u"):
        mode = "raw"
    else:
        raise ValueError(
            "scale_mode must be 'raw' or 'posterior_draw_minmax'; "
            f"got {scale_mode!r}"
        )

    if yaxis_title is None:
        yaxis_title = "Utility" if mode == "raw" else "Min-max normalized utility"

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    fig, ax = plt.subplots(figsize=figsize)

    if truth_y is not None:
        truth_y = np.asarray(truth_y, dtype=float).reshape(-1)
        if truth_y.shape[0] != g_vals.shape[0]:
            raise ValueError(
                f"truth_y has len={truth_y.shape[0]} but expected {g_vals.shape[0]} "
                f"(poster_min/max mismatch?)"
            )
        ax.plot(
            g_vals,
            truth_y,
            color=truth_color,
            linestyle=truth_dash,
            linewidth=truth_width,
            label=str(truth_label),
            zorder=5,
        )

    sorted_series = _prepare_posterior_overlay_series(series)
    for i, s in enumerate(sorted_series):
        label = str(s.get("label", f"model_{i+1}"))
        display_label = method_display_label(label)
        mean = np.asarray(s["mean"], dtype=float).reshape(-1)
        band_key = str(band_source).strip().lower()
        if band_key == "sem":
            low_in = s.get("sem_low", s.get("low", None))
            high_in = s.get("sem_high", s.get("high", None))
        elif band_key == "sd":
            low_in = s.get("sd_low", s.get("low", None))
            high_in = s.get("sd_high", s.get("high", None))
        else:
            low_in = s.get("low", None)
            high_in = s.get("high", None)

        if show_hdi_band and (low_in is None or high_in is None):
            raise ValueError(f"{label}: show_hdi_band=True requires 'low' and 'high'.")

        if low_in is None or high_in is None:
            low = mean
            high = mean
        else:
            low = np.asarray(low_in, dtype=float).reshape(-1)
            high = np.asarray(high_in, dtype=float).reshape(-1)

        if mean.shape[0] != g_vals.shape[0]:
            raise ValueError(
                f"{label}: mean has len={mean.shape[0]} but expected {g_vals.shape[0]} "
                f"(poster_min/max mismatch?)"
            )

        if mode == "raw":
            mean_plot = mean
            low_plot = low
            high_plot = high
        else:
            already_normalized = bool(
                s.get("already_normalized", False)
                or s.get("normalization_method", "") == "posterior_draw_minmax"
            )
            if already_normalized:
                mean_plot = mean
                low_plot = low
                high_plot = high
            else:
                mean_plot, low_plot, high_plot, _lo, _hi, _deg = _minmax_scale_three(mean, low, high)

        color = method_color(label, palette[i % len(palette)])
        if show_hdi_band:
            ax.fill_between(
                g_vals,
                low_plot,
                high_plot,
                color=color,
                alpha=band_alpha,
                linewidth=0,
            )
        ax.plot(
            g_vals,
            mean_plot,
            color=color,
            linewidth=line_width,
            label=display_label,
        )

    ax.set_xlabel("Poster", fontsize=13)
    ax.set_ylabel(str(yaxis_title), fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if yaxis_range is not None:
        ax.set_ylim(*list(yaxis_range))
    elif mode == "posterior_draw_minmax":
        ax.set_ylim(*_bounded_unit_range_with_headroom())

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=int(legend_ncol),
        frameon=False,
        fontsize=float(legend_fontsize),
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.32)
    fig.savefig(out_path_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_posterior_U_individual_pngs(
    series,
    *,
    poster_min: int,
    poster_max: int,
    out_dir: Path,
    scale_mode: str = "raw",
    yaxis_title: str | None = None,
    yaxis_range=None,
    band_alpha: float = 0.16,
    band_source: str = "default",
    show_hdi_band: bool = True,
    truth_y=None,
    truth_label: str = "Ground truth",
    truth_color: str = "black",
    truth_dash: str = "solid",
    truth_width: float = 2.8,
    line_width: float = 2.2,
    legend_fontsize: float = 10.5,
    legend_ncol: int = 2,
    filename_prefix: str = "thesis_posterior_U",
    figsize: tuple[float, float] = (6.4, 5.6),
    dpi: int = 300,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for s in _prepare_posterior_overlay_series(series):
        label = str(s.get("label", "posterior"))
        display_label = method_display_label(label)
        out_path = out_dir / f"{filename_prefix}_{_safe_plot_filename(display_label)}.png"
        ok = plot_posterior_U_overlay_png(
            [s],
            poster_min=poster_min,
            poster_max=poster_max,
            out_path_png=out_path,
            scale_mode=scale_mode,
            yaxis_title=yaxis_title,
            yaxis_range=yaxis_range,
            band_alpha=band_alpha,
            band_source=band_source,
            show_hdi_band=show_hdi_band,
            truth_y=truth_y,
            truth_label=truth_label,
            truth_color=truth_color,
            truth_dash=truth_dash,
            truth_width=truth_width,
            line_width=line_width,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            figsize=figsize,
            dpi=dpi,
        )
        if ok:
            written[label] = str(out_path)
    return written


def plot_posterior_U_overlay_plotly(
    series,
    *,
    poster_min: int,
    poster_max: int,
    out_path_html: Path,
    title: str = "Posterior U(g) overlay",
    scale_mode: str = "posterior_draw_minmax",
    yaxis_title: str | None = None,
    yaxis_range=None,
    hdi_prob: float | None = None,
    band_alpha: float = 0.18,
    show_hdi_band: bool = False,
    band_label: str | None = None,
    truth_y=None,                         
    truth_label: str = "Ground truth",    
    truth_color: str = "black",           
    truth_dash: str = "dash",             
    truth_width: int = 3,                 
):
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    g_vals = np.arange(int(poster_min), int(poster_max) + 1, dtype=int)
    mode = str(scale_mode).strip().lower()
    if mode in ("minmax", "posterior_minmax", "posterior_draw_minmax"):
        mode = "posterior_draw_minmax"
    elif mode in ("raw", "raw_u", "u"):
        mode = "raw"
    else:
        raise ValueError(
            "scale_mode must be 'raw' or 'posterior_draw_minmax'; "
            f"got {scale_mode!r}"
        )

    if yaxis_title is None:
        yaxis_title = (
            "Utility U(g)"
            if mode == "raw"
            else "Min-max normalized utility"
        )

    fig = go.Figure()
    palette = list(getattr(qualitative, "Plotly", [])) or [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)",
        "rgb(214,39,40)", "rgb(148,103,189)", "rgb(140,86,75)",
    ]

    if truth_y is not None:
        truth_y = np.asarray(truth_y, dtype=float).reshape(-1)
        if truth_y.shape[0] != g_vals.shape[0]:
            raise ValueError(
                f"truth_y has len={truth_y.shape[0]} but expected {g_vals.shape[0]} "
                f"(poster_min/max mismatch?)"
            )

        fig.add_trace(go.Scatter(
            x=g_vals, y=truth_y,
            mode="lines",
            name=str(truth_label),
            legendgroup=str(truth_label),
            line=dict(color=str(truth_color), width=int(truth_width), dash=str(truth_dash)),
            hovertemplate=(
                f"{truth_label}<br>"
                "g=%{x}<br>"
                "U=%{y:.4f}<extra></extra>"
            ),
        ))

    sorted_series = _prepare_posterior_overlay_series(series)

    for i, s in enumerate(sorted_series):
        label = str(s.get("label", f"model_{i+1}"))
        display_label = method_display_label(label)

        mean = np.asarray(s["mean"], dtype=float).reshape(-1)

        low_in = s.get("low", None)
        high_in = s.get("high", None)

        if show_hdi_band and (low_in is None or high_in is None):
            raise ValueError(f"{label}: show_hdi_band=True requires 'low' and 'high' in series dict.")

        if low_in is None or high_in is None:
            low = mean
            high = mean
        else:
            low = np.asarray(low_in, dtype=float).reshape(-1)
            high = np.asarray(high_in, dtype=float).reshape(-1)

        if mean.shape[0] != g_vals.shape[0]:
            raise ValueError(
                f"{label}: mean has len={mean.shape[0]} but expected {g_vals.shape[0]} "
                f"(poster_min/max mismatch?)"
            )
        if low.shape[0] != mean.shape[0] or high.shape[0] != mean.shape[0]:
            raise ValueError(
                f"{label}: low/high lengths must match mean length; "
                f"got mean={mean.shape[0]}, low={low.shape[0]}, high={high.shape[0]}"
            )

        if mode == "raw":
            mean_plot = mean
            low_plot = low
            high_plot = high
            hover_name = "U"
        else:
            already_normalized = bool(
                s.get("already_normalized", False)
                or s.get("normalization_method", "") == "posterior_draw_minmax"
            )

            if already_normalized:
                mean_plot = mean
                low_plot = low
                high_plot = high
            else:
                mean_plot, low_plot, high_plot, _lo, _hi, _deg = _minmax_scale_three(mean, low, high)
            hover_name = "U_minmax"

        color = method_color(label, palette[i % len(palette)])
        fill = _color_to_rgba(color, band_alpha)
        bound_color = _color_to_rgba(color, 0.55)

        if show_hdi_band:
            fig.add_trace(go.Scatter(
                x=g_vals, y=low_plot, mode="lines+markers",
                line=dict(width=1, color=bound_color, dash="dot"),
                marker=dict(size=3, color=bound_color),
                showlegend=False,
                legendgroup=display_label,
                hoverinfo="skip",
                name=f"{display_label} lower",
            ))

            fig.add_trace(go.Scatter(
                x=g_vals, y=high_plot, mode="lines+markers",
                fill="tonexty", fillcolor=fill,
                line=dict(width=1, color=bound_color, dash="dot"),
                marker=dict(size=3, color=bound_color),
                showlegend=False,
                legendgroup=display_label,
                hoverinfo="skip",
                name=f"{display_label} upper",
            ))

        fig.add_trace(go.Scatter(
            x=g_vals, y=mean_plot, mode="lines",
            line=dict(width=2, color=color),
            name=display_label,
            legendgroup=display_label,
            hovertemplate=(
                f"{display_label}<br>"
                "g=%{x}<br>"
                f"{hover_name}=%{{y:.4f}}<extra></extra>"
            ),
        ))

    hdi_txt = ""
    if show_hdi_band and (hdi_prob is not None):
        hdi_txt = f" ({int(round(float(hdi_prob) * 100))}% HDI)"
    elif show_hdi_band and band_label:
        hdi_txt = f" ({band_label})"

    title_text = "" if title is None else str(title)
    if title_text and hdi_txt:
        title_text = title_text + hdi_txt

    fig.update_layout(
        title=title_text,
        xaxis_title="Poster",
        yaxis_title=str(yaxis_title),
        template="plotly_white",
        margin=dict(l=70, r=30, t=90 if title_text else 45, b=125),
        height=560,
        legend=dict(
            orientation="h",
            x=0.0,
            y=-0.22,
            xanchor="left",
            yanchor="top",
        ),
    )
    if yaxis_range is not None:
        fig.update_yaxes(range=list(yaxis_range))
    elif mode == "posterior_draw_minmax":
        fig.update_yaxes(range=_bounded_unit_range_with_headroom())

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig


def plot_posterior_U_minmax_overlay_plotly(
    series,
    *,
    poster_min: int,
    poster_max: int,
    out_path_html: Path,
    title: str = "Posterior of min-max normalized U(g)",
    hdi_prob: float | None = None,
    band_alpha: float = 0.18,
    show_hdi_band: bool = False,
    truth_y=None,
    truth_label: str = "Ground truth",
    truth_color: str = "black",
    truth_dash: str = "dash",
    truth_width: int = 3,
):
    return plot_posterior_U_overlay_plotly(
        series,
        poster_min=poster_min,
        poster_max=poster_max,
        out_path_html=out_path_html,
        title=title,
        scale_mode="posterior_draw_minmax",
        yaxis_title="Min-max normalized utility",
        yaxis_range=_bounded_unit_range_with_headroom(),
        hdi_prob=hdi_prob,
        band_alpha=band_alpha,
        show_hdi_band=show_hdi_band,
        truth_y=truth_y,
        truth_label=truth_label,
        truth_color=truth_color,
        truth_dash=truth_dash,
        truth_width=truth_width,
    )


def plot_prefix_metrics_learning_curves_multi_plotly(
    prefix_summaries_by_label,
    *,
    out_path_html: Path,
    title: str = "RMSE, interval coverage, Gaussian NLPD, pairwise accuracy, and simple regret (overlay)",
    use_metrics: str = "minmax",   # "minmax" or "raw"
    include_gaussian_nlpd: bool = True,
    include_pairwise_acc: bool = True,
    include_recommendation_metrics: bool = True,
    include_2d_metrics: bool = True,
):
    """
    prefix_summaries_by_label: dict[label -> materialized_prefix_summaries_list]
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.colors import qualitative
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    if not isinstance(prefix_summaries_by_label, dict) or len(prefix_summaries_by_label) == 0:
        raise ValueError("plot_prefix_metrics_learning_curves_multi_plotly: expected non-empty dict[label -> prefixes].")

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)

    def _get_metric(s, key: str):
        if key in s:
            return s.get(key)

        m = s.get("metrics", {})
        if not isinstance(m, dict):
            return None

        if use_metrics in m and isinstance(m[use_metrics], dict) and key in m[use_metrics]:
            return m[use_metrics].get(key)

        mm = m.get("minmax", {})
        if isinstance(mm, dict) and key in mm:
            return mm.get(key)
        raw = m.get("raw", {})
        if isinstance(raw, dict) and key in raw:
            return raw.get(key)
        return None

    def _as_float(x):
        try:
            if x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _get_pairwise_acc(s):
        val = _as_float(_get_metric(s, "posterior_pairwise_acc_comp"))
        if np.isnan(val):
            val = _as_float(_get_metric(s, "pairwise_acc_comp"))
        return val

    has_2d_metrics = False
    if include_2d_metrics:
        for summaries in prefix_summaries_by_label.values():
            for s in summaries or []:
                if np.isfinite(_as_float(_get_metric(s, "contrast_error"))) or np.isfinite(
                    _as_float(_get_metric(s, "contrast_profile_rmse"))
                ):
                    has_2d_metrics = True
                    break
            if has_2d_metrics:
                break

    #labels = sorted(prefix_summaries_by_label.keys())
    def _label_str(x) -> str:
        return x.as_posix() if isinstance(x, Path) else str(x)

    labels = sorted(prefix_summaries_by_label.keys(), key=lambda label: method_sort_key(_label_str(label)))

    subplot_titles = [_metric_label("rmse"), _metric_label("interval_coverage")]
    if include_gaussian_nlpd:
        subplot_titles.append(_metric_label("gaussian_nlpd_raw"))
    if include_pairwise_acc:
        subplot_titles.append(_metric_label("posterior_pairwise_acc_comp"))
    if include_recommendation_metrics:
        subplot_titles.append(_metric_label("simple_regret"))
    if has_2d_metrics:
        subplot_titles.extend([
            _metric_label("contrast_error"),
            _metric_label("contrast_profile_rmse"),
        ])
    n_rows = len(subplot_titles)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=subplot_titles,
    )

    palette = list(getattr(qualitative, "Plotly", [])) or [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)",
        "rgb(214,39,40)", "rgb(148,103,189)", "rgb(140,86,75)",
    ]
    all_x_values = []

    for i, label in enumerate(labels):
        label_s = _label_str(label)
        display_label = method_display_label(label_s)
        
        prefix_summaries = prefix_summaries_by_label[label]
        if not prefix_summaries:
            continue

        xs, ks, n_obs, n_pbo, n_ties = [], [], [], [], []
        rmse, gaussian_nlpd, cov, pairacc, regret = [], [], [], [], []
        contrast_error, contrast_profile_rmse = [], []

        for j, s in enumerate(prefix_summaries):
            ks.append(int(s.get("k", j + 1)))
            n_obs.append(int(s.get("n_obs", 0)))
            n_pbo.append(int(s.get("n_pbo_duels", 0)))
            n_ties.append(int(s.get("n_ties_dropped", 0)))

            x = s.get("k_traj", None)
            if x is None:
                x = s.get("n_traj", None)
            if x is None:
                x = s.get("k", j + 1)
            xs.append(int(x))

            rmse.append(_as_float(_get_metric(s, "rmse")))
            gaussian_nlpd.append(_as_float(_get_metric(s, "gaussian_nlpd_raw")))
            cov.append(_as_float(_get_metric(s, "interval_coverage")))
            pairacc.append(_get_pairwise_acc(s))
            regret.append(_as_float(_get_metric(s, "simple_regret")))
            contrast_error.append(_as_float(_get_metric(s, "contrast_error")))
            contrast_profile_rmse.append(_as_float(_get_metric(s, "contrast_profile_rmse")))

        xs = np.asarray(xs, dtype=int)
        all_x_values.extend(xs.astype(float).tolist())
        rmse = np.asarray(rmse, dtype=float)
        gaussian_nlpd = np.asarray(gaussian_nlpd, dtype=float)
        cov = np.asarray(cov, dtype=float)
        pairacc = np.asarray(pairacc, dtype=float)
        regret = np.asarray(regret, dtype=float)
        contrast_error = np.asarray(contrast_error, dtype=float)
        contrast_profile_rmse = np.asarray(contrast_profile_rmse, dtype=float)
        custom = np.stack(
            [
                np.asarray(ks, dtype=int),
                np.asarray(n_obs, dtype=int),
                np.asarray(n_pbo, dtype=int),
                np.asarray(n_ties, dtype=int),
            ],
            axis=1,
        )

        color = method_color(label_s, palette[i % len(palette)])

        fig.add_trace(
            go.Scatter(
                x=xs, y=rmse,
                mode="lines+markers",
                name=display_label,
                legendgroup=display_label,
                showlegend=True,
                line=dict(color=color),
                marker=dict(size=7, color=color),
                hovertemplate=(
                    f"{display_label}<br>"
                    "k_traj=%{x}<br>"
                    "rmse=%{y:.6f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=xs, y=cov,
                mode="lines+markers",
                name=display_label,
                legendgroup=display_label,
                showlegend=False,
                line=dict(color=color),
                marker=dict(size=7, color=color),
                hovertemplate=(
                    f"{display_label}<br>"
                    "k_traj=%{x}<br>"
                    "coverage=%{y:.4f}<br>"
                    "k=%{customdata[0]}<br>"
                    "n_obs=%{customdata[1]}<br>"
                    "n_pbo_duels=%{customdata[2]}<br>"
                    "n_ties_dropped=%{customdata[3]}<extra></extra>"
                ),
                customdata=custom,
            ),
            row=2, col=1
        )

        next_row = 3
        if include_gaussian_nlpd:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=gaussian_nlpd,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=False,
                    line=dict(color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{display_label}<br>"
                        "k_traj=%{x}<br>"
                        "Gaussian NLPD=%{y:.6f}<br>"
                        "k=%{customdata[0]}<br>"
                        "n_obs=%{customdata[1]}<br>"
                        "n_pbo_duels=%{customdata[2]}<br>"
                        "n_ties_dropped=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=custom,
                ),
                row=next_row, col=1
            )
            next_row += 1

        if include_pairwise_acc:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=pairacc,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=False,
                    line=dict(color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{display_label}<br>"
                        "k_traj=%{x}<br>"
                        "posterior_pairwise_acc=%{y:.4f}<br>"
                        "k=%{customdata[0]}<br>"
                        "n_obs=%{customdata[1]}<br>"
                        "n_pbo_duels=%{customdata[2]}<br>"
                        "n_ties_dropped=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=custom,
                ),
                row=next_row, col=1
            )
            next_row += 1

        if include_recommendation_metrics:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=regret,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=False,
                    line=dict(color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{display_label}<br>"
                        "k_traj=%{x}<br>"
                        "simple_regret=%{y:.4f}<br>"
                        "k=%{customdata[0]}<br>"
                        "n_obs=%{customdata[1]}<br>"
                        "n_pbo_duels=%{customdata[2]}<br>"
                        "n_ties_dropped=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=custom,
                ),
                row=next_row, col=1
            )
            next_row += 1

        if has_2d_metrics:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=contrast_error,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=False,
                    line=dict(color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{display_label}<br>"
                        "k_traj=%{x}<br>"
                        "contrast_error=%{y:.4f}<br>"
                        "k=%{customdata[0]}<br>"
                        "n_obs=%{customdata[1]}<br>"
                        "n_pbo_duels=%{customdata[2]}<br>"
                        "n_ties_dropped=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=custom,
                ),
                row=next_row,
                col=1,
            )
            next_row += 1

            fig.add_trace(
                go.Scatter(
                    x=xs, y=contrast_profile_rmse,
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=False,
                    line=dict(color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{display_label}<br>"
                        "k_traj=%{x}<br>"
                        "contrast_profile_rmse=%{y:.6f}<br>"
                        "k=%{customdata[0]}<br>"
                        "n_obs=%{customdata[1]}<br>"
                        "n_pbo_duels=%{customdata[2]}<br>"
                        "n_ties_dropped=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=custom,
                ),
                row=next_row,
                col=1,
            )

    fig.update_yaxes(title_text=_metric_label("rmse"), row=1, col=1)
    fig.update_yaxes(
        title_text=_metric_label("interval_coverage"),
        range=_bounded_unit_range_with_headroom(),
        row=2,
        col=1,
    )
    next_row = 3
    if include_gaussian_nlpd:
        fig.update_yaxes(title_text=_metric_label("gaussian_nlpd_raw"), row=next_row, col=1)
        next_row += 1
    if include_pairwise_acc:
        fig.update_yaxes(
            title_text=_metric_label("posterior_pairwise_acc_comp"),
            range=_bounded_unit_range_with_headroom(),
            row=next_row,
            col=1,
        )
        next_row += 1
    if include_recommendation_metrics:
        fig.update_yaxes(title_text=_metric_label("simple_regret"), row=next_row, col=1)
        next_row += 1
    if has_2d_metrics:
        fig.update_yaxes(title_text=_metric_label("contrast_error"), row=next_row, col=1)
        next_row += 1
        fig.update_yaxes(title_text=_metric_label("contrast_profile_rmse"), row=next_row, col=1)

    _show_x_axis_on_all_rows(
        fig,
        n_rows=n_rows,
        title_text="Trajectory budget",
        x_values=all_x_values,
    )

    title_text = "" if title is None else str(title)

    fig.update_layout(
        title=dict(text=title_text, x=0.01, y=0.985, xanchor="left", yanchor="top"),
        template="plotly_white",
        height=max(700, 280 * n_rows),
        margin=dict(l=70, r=30, t=110 if title_text else 85, b=70),
        legend=dict(
            orientation="h",
            xanchor="right",
            x=1.0,
            yanchor="bottom",
            y=1.02,
        ),
    )

    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig


def _grid_2d(values, *, grid_k: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    k = int(grid_k)
    if arr.shape[0] != k * k:
        raise ValueError(f"Expected {k*k} values for grid_k={k}, got {arr.shape[0]}.")
    return arr.reshape((k, k))


def plot_2d_posterior_heatmaps_plotly(
    *,
    U_true,
    U_post_mean,
    U_post_sd,
    grid_k: int,
    out_path_html: Path,
    title: str = "2D posterior utility diagnostics",
    recommendation: dict | None = None,
    zmin=None,
    zmax=None,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    k = int(grid_k)
    x = np.arange(1, k + 1, dtype=int)
    y = np.arange(1, k + 1, dtype=int)
    true_grid = _grid_2d(U_true, grid_k=k)
    mean_grid = _grid_2d(U_post_mean, grid_k=k)
    sd_grid = _grid_2d(U_post_sd, grid_k=k)
    err_grid = np.abs(mean_grid - true_grid)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "True utility",
            "Posterior mean",
            "Posterior SD",
            "Absolute error",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    panels = [
        (true_grid, 1, 1, "U true", "coloraxis"),
        (mean_grid, 1, 2, "mean U", "coloraxis"),
        (sd_grid, 2, 1, "SD", "coloraxis2"),
        (err_grid, 2, 2, "|error|", "coloraxis3"),
    ]
    for z, row, col, label, coloraxis in panels:
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                coloraxis=coloraxis,
                hovertemplate=(
                    "background=%{x}<br>"
                    "headline=%{y}<br>"
                    f"{label}=%{{z:.4f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    if recommendation:
        markers = []
        if recommendation.get("h_star") and recommendation.get("b_star"):
            markers.append(
                (
                    int(recommendation["b_star"]),
                    int(recommendation["h_star"]),
                    "true optimum",
                    "star",
                    "black",
                )
            )
        if recommendation.get("h_hat") and recommendation.get("b_hat"):
            markers.append(
                (
                    int(recommendation["b_hat"]),
                    int(recommendation["h_hat"]),
                    "recommendation",
                    "x",
                    "crimson",
                )
            )
        for bx, hy, name, symbol, color in markers:
            for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
                fig.add_trace(
                    go.Scatter(
                        x=[bx],
                        y=[hy],
                        mode="markers",
                        marker=dict(symbol=symbol, size=13, color=color, line=dict(width=1)),
                        name=name,
                        showlegend=(row == 1 and col == 1),
                        hovertemplate=f"{name}<br>background=%{{x}}<br>headline=%{{y}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="Background grayscale", row=row, col=col)
            fig.update_yaxes(title_text="Headline grayscale", row=row, col=col)

    def _coloraxis(title_text, *, y, length, lo=None, hi=None):
        cfg = dict(
            colorscale="Viridis",
            colorbar=dict(
                title=title_text,
                x=1.015,
                y=float(y),
                len=float(length),
                thickness=14,
                outlinewidth=0,
            ),
        )
        if lo is not None:
            cfg["cmin"] = lo
        if hi is not None:
            cfg["cmax"] = hi
        return cfg

    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.985, xanchor="left", yanchor="top"),
        template="plotly_white",
        height=830,
        margin=dict(l=70, r=135, t=165, b=105),
        coloraxis=_coloraxis("Utility", y=0.79, length=0.36, lo=zmin, hi=zmax),
        coloraxis2=_coloraxis("Posterior SD", y=0.39, length=0.24),
        coloraxis3=_coloraxis("Absolute error", y=0.13, length=0.24),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.07,
            yanchor="top",
        ),
    )

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig


def plot_2d_prefix_heatmap_slider_plotly(
    prefix_summaries,
    *,
    grid_k: int,
    out_path_html: Path,
    title: str = "2D prefix posterior mean utility",
    value_key: str = "mean",
    zmin=None,
    zmax=None,
):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    if not prefix_summaries:
        raise ValueError("prefix_summaries is empty.")

    k = int(grid_k)
    coords = np.arange(1, k + 1, dtype=int)
    fig = go.Figure()

    for i, s in enumerate(prefix_summaries):
        values = np.asarray(s[value_key], dtype=float).reshape(-1)
        fig.add_trace(
            go.Heatmap(
                x=coords,
                y=coords,
                z=_grid_2d(values, grid_k=k),
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                visible=(i == 0),
                colorbar=dict(title=value_key),
                hovertemplate=(
                    "background=%{x}<br>"
                    "headline=%{y}<br>"
                    f"{value_key}=%{{z:.4f}}<extra></extra>"
                ),
            )
        )

    steps = []
    for i, s in enumerate(prefix_summaries):
        vis = [False] * len(prefix_summaries)
        vis[i] = True
        steps.append(
            dict(
                method="update",
                label=f"{s.get('k_traj', s.get('k', i + 1))}",
                args=[
                    {"visible": vis},
                    {
                        "title": (
                            f"{title}<br><sup>"
                            f"k={s.get('k', i + 1)}, "
                            f"k_traj={s.get('k_traj', '')}, "
                            f"n={s.get('n_obs', '')}</sup>"
                        )
                    },
                ],
            )
        )

    first = prefix_summaries[0]
    fig.update_layout(
        title=(
            f"{title}<br><sup>k={first.get('k', 1)}, "
            f"k_traj={first.get('k_traj', '')}, n={first.get('n_obs', '')}</sup>"
        ),
        xaxis_title="Background grayscale",
        yaxis_title="Headline grayscale",
        sliders=[dict(active=0, steps=steps, x=0.05, xanchor="left", y=0.0, yanchor="top")],
        template="plotly_white",
        height=650,
        margin=dict(l=70, r=40, t=100, b=90),
    )

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig


def plot_contrast_profile_plotly(
    profile_rows,
    *,
    out_path_html: Path,
    title: str = "Contrast profile recovery",
    preferred_contrast: int | float | None = None,
):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly required: pip install plotly") from e

    rows = list(profile_rows)
    x = np.asarray([r["contrast"] for r in rows], dtype=int)
    true_y = np.asarray([r["U_true_profile"] for r in rows], dtype=float)
    mean_y = np.asarray([r["U_post_mean_profile"] for r in rows], dtype=float)
    low_y = np.asarray([r.get("U_hdi_low_profile", np.nan) for r in rows], dtype=float)
    high_y = np.asarray([r.get("U_hdi_high_profile", np.nan) for r in rows], dtype=float)

    fig = go.Figure()
    if np.any(np.isfinite(low_y)) and np.any(np.isfinite(high_y)):
        fig.add_trace(go.Scatter(x=x, y=low_y, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=high_y,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.18)",
                line=dict(width=0),
                name="Posterior interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(x=x, y=mean_y, mode="lines+markers", name="Posterior mean")
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=true_y,
            mode="lines+markers",
            name="Ground truth",
            line=dict(color="black", width=3),
        )
    )
    if preferred_contrast is not None:
        fig.add_vline(
            x=float(preferred_contrast),
            line_dash="dash",
            line_color="crimson",
            annotation_text="c*",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Contrast |h-b|",
        yaxis_title="Mean utility by contrast",
        template="plotly_white",
        height=500,
        margin=dict(l=70, r=30, t=80, b=60),
    )

    out_path_html = Path(out_path_html)
    out_path_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    return fig
