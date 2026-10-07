from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from io_utils import save_jsonl
from poster_env import DEC, EXPORT, G_MAX, G_MIN, INC, Problem1D
from sweep_runner import run_trajectory_from_start
from user_models import UserModel4_EndpointDiscountedBoltzmann, _softmax_stable
from utility import VALID_UTILITY_KINDS, get_utility, get_utility_label


DEFAULT_BALANCED_CONFIGS = ((0.95, 15.0), (0.99, 10.0))
DEFAULT_CLIPPED_CONFIGS = ((0.95, 15.0),)
DEFAULT_REPS = (3, 5)
DEFAULT_PEAK_CENTERS = (20, 85)
DEFAULT_EXAMPLE_STARTS = (1, 10, 20, 35, 50, 70, 85, 100)


def _stable_seed(*items: object, base_seed: int) -> int:
    payload = "|".join(str(x) for x in (base_seed, *items)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _format_number(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(float(x)).rstrip("0").rstrip(".")


def _parse_float_pairs(text: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected gamma:alpha pair, got {item!r}")
        gamma_s, alpha_s = item.split(":", 1)
        pairs.append((float(gamma_s), float(alpha_s)))
    return tuple(pairs)


def _parse_ints(text: str) -> tuple[int, ...]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return tuple(vals)


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_basin_states(
    centers: tuple[int, ...],
    radius: int,
    g_min: int = G_MIN,
    g_max: int = G_MAX,
) -> set[int]:
    states: set[int] = set()
    for center in centers:
        lo = max(g_min, int(center) - radius)
        hi = min(g_max, int(center) + radius)
        states.update(range(lo, hi + 1))
    return states


def _balanced_start_counts(reps_per_start: int) -> dict[int, int]:
    return {g: int(reps_per_start) for g in range(G_MIN, G_MAX + 1)}


def _even_extra_indices(n: int, r: int) -> set[int]:
    if r <= 0:
        return set()
    if r >= n:
        return set(range(n))
    return {int(round(x)) for x in np.linspace(0, n - 1, r)}


def _peak_clipped_start_counts(
    equivalent_reps_per_state: int,
    peak_basin: set[int],
) -> dict[int, int]:
    """Keep every start state, but cap repetitions in peak-basin states."""
    states = list(range(G_MIN, G_MAX + 1))
    total_target = len(states) * int(equivalent_reps_per_state)
    if total_target < len(states):
        raise ValueError("equivalent_reps_per_state must be at least 1")

    counts = {g: 1 for g in states}
    non_peak = [g for g in states if g not in peak_basin]
    remaining = total_target - len(states)
    q, r = divmod(remaining, len(non_peak))
    extra_once = _even_extra_indices(len(non_peak), r)

    for i, g in enumerate(non_peak):
        counts[g] += q
        if i in extra_once:
            counts[g] += 1
    return counts


def _policy_probs(user: UserModel4_EndpointDiscountedBoltzmann, g: int) -> dict[str, float]:
    actions = [EXPORT]
    if g < user.g_max:
        actions.append(INC)
    if g > user.g_min:
        actions.append(DEC)
    i = int(g) - user.g_min
    qvals = np.array([user.Q[a][i] for a in actions], dtype=float)
    probs = _softmax_stable(user.alpha * qvals)
    return {a: float(p) for a, p in zip(actions, probs)}


def _expected_start_policy_stats(
    user: UserModel4_EndpointDiscountedBoltzmann,
    start_counts: dict[int, int],
    peak_basin: set[int],
) -> dict[str, float | int]:
    total = sum(start_counts.values())
    expected_imm = 0.0
    expected_imm_peak = 0.0
    for g, count in start_counts.items():
        p_export = _policy_probs(user, g).get(EXPORT, 0.0)
        expected_imm += count * p_export
        if g in peak_basin:
            expected_imm_peak += count * p_export

    det_export_states = 0
    for g in range(user.g_min, user.g_max + 1):
        i = g - user.g_min
        q_export = user.Q[EXPORT][i]
        q_inc = user.Q[INC][i]
        q_dec = user.Q[DEC][i]
        if q_export >= max(q_inc, q_dec):
            det_export_states += 1

    return {
        "det_export_states": int(det_export_states),
        "expected_immediate_export": float(expected_imm),
        "expected_immediate_export_frac": float(expected_imm / total),
        "expected_immediate_peak_basin": float(expected_imm_peak),
        "expected_immediate_peak_basin_frac_all": float(expected_imm_peak / total),
        "expected_immediate_peak_basin_share": (
            float(expected_imm_peak / expected_imm) if expected_imm > 0 else float("nan")
        ),
    }


def _entropy_norm(counts: np.ndarray, support_size: int) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan")
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(support_size) if support_size > 1 else float("nan")


def _summarize_start_counts(start_counts: dict[int, int]) -> dict[str, float | int]:
    counts = np.array([start_counts[g] for g in range(G_MIN, G_MAX + 1)], dtype=int)
    total = int(np.sum(counts))
    top5 = np.sort(counts)[-5:]
    top10 = np.sort(counts)[-10:]
    return {
        "n_start_states": int(np.sum(counts > 0)),
        "start_count_min": int(np.min(counts)),
        "start_count_max": int(np.max(counts)),
        "start_count_mean": float(np.mean(counts)),
        "start_entropy_norm": _entropy_norm(counts, support_size=len(counts)),
        "start_top5_share": float(np.sum(top5) / total),
        "start_top10_share": float(np.sum(top10) / total),
    }


def _summarize_dataset(
    per_traj_rows: list[dict],
    start_counts: dict[int, int],
    user: UserModel4_EndpointDiscountedBoltzmann,
    peak_basin: set[int],
) -> dict[str, float | int]:
    n = len(per_traj_rows)
    starts = np.array([int(r["start_poster"]) for r in per_traj_rows], dtype=int)
    exports = np.array([int(r["export_poster"]) for r in per_traj_rows], dtype=int)
    steps = np.array([int(r["steps"]) for r in per_traj_rows], dtype=int)

    immediate = steps == 1
    start_eq = starts == exports
    export_counts = np.zeros(G_MAX - G_MIN + 1, dtype=int)
    for g in exports:
        export_counts[g - G_MIN] += 1

    sorted_counts = np.sort(export_counts)
    n_immediate = int(np.sum(immediate))
    n_immediate_peak = int(
        np.sum(immediate & np.array([int(s) in peak_basin for s in starts], dtype=bool))
    )
    n_exports_peak = int(np.sum([export_counts[g - G_MIN] for g in peak_basin]))

    summary: dict[str, float | int] = {
        "n_traj": int(n),
        "n_rows": int(np.sum(steps)),
        "mean_rows_per_traj": float(np.mean(steps)),
        "median_rows_per_traj": float(np.median(steps)),
        "max_rows_per_traj": int(np.max(steps)),
        "n_immediate_export": n_immediate,
        "frac_immediate_export": float(n_immediate / n),
        "n_start_eq_export": int(np.sum(start_eq)),
        "frac_start_eq_export": float(np.mean(start_eq)),
        "n_nontrivial_pbo_duels": int(np.sum(~start_eq)),
        "frac_nontrivial_pbo_duels": float(np.mean(~start_eq)),
        "n_unique_export_states": int(np.sum(export_counts > 0)),
        "export_entropy_norm": _entropy_norm(export_counts, support_size=len(export_counts)),
        "export_top1_share": float(sorted_counts[-1] / n),
        "export_top5_share": float(np.sum(sorted_counts[-5:]) / n),
        "export_top10_share": float(np.sum(sorted_counts[-10:]) / n),
        "n_immediate_peak_basin": n_immediate_peak,
        "frac_immediate_peak_basin_all": float(n_immediate_peak / n),
        "frac_immediate_peak_basin_among_immediate": (
            float(n_immediate_peak / n_immediate) if n_immediate else float("nan")
        ),
        "n_exports_peak_basin": n_exports_peak,
        "frac_exports_peak_basin": float(n_exports_peak / n),
    }
    summary.update(_summarize_start_counts(start_counts))
    summary.update(_expected_start_policy_stats(user, start_counts, peak_basin))
    return summary


def _per_start_rows(per_traj_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for start in range(G_MIN, G_MAX + 1):
        group = [r for r in per_traj_rows if int(r["start_poster"]) == start]
        if not group:
            continue
        exports = np.array([int(r["export_poster"]) for r in group], dtype=float)
        steps = np.array([int(r["steps"]) for r in group], dtype=float)
        utils = np.array([float(r["export_utility"]) for r in group], dtype=float)
        rows.append(
            {
                "start_poster": int(start),
                "n": int(len(group)),
                "export_poster_mean": float(np.mean(exports)),
                "export_poster_std": float(np.std(exports)),
                "steps_mean": float(np.mean(steps)),
                "steps_std": float(np.std(steps)),
                "export_utility_mean": float(np.mean(utils)),
                "export_utility_std": float(np.std(utils)),
                "n_immediate_export": int(np.sum(steps == 1)),
                "frac_immediate_export": float(np.mean(steps == 1)),
            }
        )
    return rows


def _export_count_rows(per_traj_rows: list[dict], dataset_label: str) -> list[dict]:
    counts = {g: 0 for g in range(G_MIN, G_MAX + 1)}
    for row in per_traj_rows:
        counts[int(row["export_poster"])] += 1
    return [
        {
            "dataset": dataset_label,
            "poster": int(g),
            "export_count": int(counts[g]),
        }
        for g in range(G_MIN, G_MAX + 1)
    ]


def _start_count_rows(start_counts: dict[int, int], dataset_label: str) -> list[dict]:
    return [
        {
            "dataset": dataset_label,
            "poster": int(g),
            "start_count": int(start_counts[g]),
        }
        for g in range(G_MIN, G_MAX + 1)
    ]


def _make_label(
    utility_kind: str,
    gamma: float,
    alpha: float,
    equivalent_reps: int,
    schedule_kind: str,
) -> str:
    prefix = "" if utility_kind == "smooth" else "Jagged_"
    gamma_s = _format_number(gamma)
    alpha_s = _format_number(alpha)
    return (
        f"{prefix}UserModel4Boltzmann_gamma{gamma_s}_alpha{alpha_s}"
        f"_reps{equivalent_reps}_{schedule_kind}_train"
    )


def _generate_dataset(
    *,
    utility_kind: str,
    gamma: float,
    alpha: float,
    equivalent_reps: int,
    schedule_kind: str,
    start_counts: dict[int, int],
    out_dir: Path,
    max_steps: int,
    base_seed: int,
    shuffle: bool,
    peak_basin: set[int],
    make_plots: bool,
) -> dict:
    utility_fn = get_utility(utility_kind)
    problem = Problem1D(g_min=G_MIN, g_max=G_MAX, g0=G_MIN)
    label = _make_label(utility_kind, gamma, alpha, equivalent_reps, schedule_kind)

    user_seed = _stable_seed(label, "user", base_seed=base_seed)
    shuffle_seed = _stable_seed(label, "shuffle", base_seed=base_seed)
    user = UserModel4_EndpointDiscountedBoltzmann(
        utility_fn=utility_fn,
        g_min=G_MIN,
        g_max=G_MAX,
        gamma=float(gamma),
        alpha=float(alpha),
        rng=np.random.default_rng(user_seed),
    )

    trajectories: list[dict] = []
    for start in range(G_MIN, G_MAX + 1):
        for rep in range(int(start_counts[start])):
            traj, summary = run_trajectory_from_start(
                problem=problem,
                user=user,
                start_poster=start,
                utility_fn=utility_fn,
                max_steps=max_steps,
            )
            summary = dict(summary)
            summary["rep"] = int(rep)
            trajectories.append(
                {
                    "start_poster": int(start),
                    "rep": int(rep),
                    "traj": traj,
                    "summary": summary,
                }
            )

    if shuffle:
        np.random.default_rng(shuffle_seed).shuffle(trajectories)

    step_records: list[dict] = []
    per_traj_rows: list[dict] = []
    for traj_id, item in enumerate(trajectories):
        start_poster = int(item["start_poster"])
        rep = int(item["rep"])
        for r in item["traj"]:
            step_records.append(
                {
                    "traj_id": int(traj_id),
                    "t": int(r["t"]),
                    "poster": int(r["poster"]),
                    "action": str(r["action"]),
                    "start_poster": start_poster,
                    "rep": rep,
                    "model": label,
                }
            )
        summary = dict(item["summary"])
        summary["traj_id"] = int(traj_id)
        summary["model"] = label
        per_traj_rows.append(summary)

    data_dir = out_dir / "data"
    summary_dir = out_dir / "summaries"
    figure_dir = out_dir / "figures"
    for d in (data_dir, summary_dir, figure_dir):
        d.mkdir(parents=True, exist_ok=True)

    steps_path = data_dir / f"steps_{label}.jsonl"
    save_jsonl(str(steps_path), step_records)

    per_start = _per_start_rows(per_traj_rows)
    per_traj_path = summary_dir / f"summary_per_traj_{label}.csv"
    per_start_path = summary_dir / f"summary_per_start_{label}.csv"
    start_counts_path = summary_dir / f"start_counts_{label}.csv"
    export_counts_path = summary_dir / f"export_counts_{label}.csv"

    _save_csv(per_traj_path, per_traj_rows)
    _save_csv(per_start_path, per_start)
    _save_csv(start_counts_path, _start_count_rows(start_counts, label))
    _save_csv(export_counts_path, _export_count_rows(per_traj_rows, label))

    diagnostics = _summarize_dataset(per_traj_rows, start_counts, user, peak_basin)
    diagnostics.update(
        {
            "dataset": label,
            "utility_kind": utility_kind,
            "utility_label": get_utility_label(utility_kind),
            "gamma": float(gamma),
            "alpha": float(alpha),
            "equivalent_reps_per_state": int(equivalent_reps),
            "start_schedule": schedule_kind,
            "steps_jsonl": str(steps_path),
            "summary_per_traj_csv": str(per_traj_path),
            "summary_per_start_csv": str(per_start_path),
            "start_counts_csv": str(start_counts_path),
            "export_counts_csv": str(export_counts_path),
            "sha256": _sha256_file(steps_path),
            "user_seed": int(user_seed),
            "shuffle_seed": int(shuffle_seed),
            "shuffled_trajectories": bool(shuffle),
        }
    )

    plot_paths = {}
    if make_plots:
        plot_paths = _write_plots(
            utility_fn=utility_fn,
            utility_kind=utility_kind,
            user=user,
            label=label,
            trajectories=trajectories,
            per_traj_rows=per_traj_rows,
            per_start_rows=per_start,
            start_counts=start_counts,
            figure_dir=figure_dir,
        )
    diagnostics["plots"] = plot_paths

    return {
        "label": label,
        "diagnostics": diagnostics,
        "export_counts": _export_count_rows(per_traj_rows, label),
        "start_counts": _start_count_rows(start_counts, label),
    }


def _write_plots(
    *,
    utility_fn: Callable,
    utility_kind: str,
    user: UserModel4_EndpointDiscountedBoltzmann,
    label: str,
    trajectories: list[dict],
    per_traj_rows: list[dict],
    per_start_rows: list[dict],
    start_counts: dict[int, int],
    figure_dir: Path,
) -> dict[str, str]:
    plotly_paths = _write_plotly_plots(
        utility_fn=utility_fn,
        utility_kind=utility_kind,
        user=user,
        label=label,
        trajectories=trajectories,
        per_traj_rows=per_traj_rows,
        per_start_rows=per_start_rows,
        start_counts=start_counts,
        figure_dir=figure_dir,
    )
    if plotly_paths:
        return plotly_paths

    try:
        import matplotlib.pyplot as plt

        from plotting import (
            plot_example_trajectories,
            plot_q_and_policy,
            plot_start_to_export,
            plot_steps_vs_start,
            plot_utility_landscape,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        print(f"[plots skipped] Could not import Matplotlib/plotting: {exc}")
        return {}

    paths = {
        "utility_landscape": figure_dir / f"utility_landscape_{utility_kind}.png",
        "q_policy": figure_dir / f"q_policy_{label}.png",
        "export_counts": figure_dir / f"export_counts_{label}.png",
        "start_counts": figure_dir / f"start_counts_{label}.png",
        "steps_by_start": figure_dir / f"steps_by_start_{label}.png",
        "start_to_export": figure_dir / f"start_to_export_{label}.png",
        "example_trajectories": figure_dir / f"example_trajectories_{label}.png",
    }

    if not paths["utility_landscape"].exists():
        plot_utility_landscape(
            utility_fn,
            G_MIN,
            G_MAX,
            title=f"{get_utility_label(utility_kind)} landscape",
            save_path=str(paths["utility_landscape"]),
            show=False,
        )

    plot_q_and_policy(
        user,
        title=f"{label}: Q* values and Boltzmann policy",
        save_path=str(paths["q_policy"]),
        show=False,
    )
    plot_steps_vs_start(
        per_start_rows,
        title=f"{label}: trajectory rows by start",
        save_path=str(paths["steps_by_start"]),
        show=False,
        with_band=True,
    )
    plot_start_to_export(
        per_start_rows,
        title=f"{label}: exported poster by start",
        save_path=str(paths["start_to_export"]),
        show=False,
        with_band=True,
    )
    plot_example_trajectories(
        utility_fn,
        trajectories,
        starts_to_plot=DEFAULT_EXAMPLE_STARTS,
        title=f"{label}: example trajectories",
        save_path=str(paths["example_trajectories"]),
        show=False,
    )
    _plot_export_counts(
        utility_fn=utility_fn,
        per_traj_rows=per_traj_rows,
        title=f"{label}: export-count distribution",
        save_path=paths["export_counts"],
        plt=plt,
    )
    _plot_start_counts(
        start_counts=start_counts,
        title=f"{label}: start-count schedule",
        save_path=paths["start_counts"],
        plt=plt,
    )
    return {key: str(path) for key, path in paths.items()}


def _write_plotly_plots(
    *,
    utility_fn: Callable,
    utility_kind: str,
    user: UserModel4_EndpointDiscountedBoltzmann,
    label: str,
    trajectories: list[dict],
    per_traj_rows: list[dict],
    per_start_rows: list[dict],
    start_counts: dict[int, int],
    figure_dir: Path,
) -> dict[str, str]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        print(f"[plotly skipped] Could not import Plotly: {exc}")
        return {}

    paths = {
        "utility_landscape": figure_dir / f"utility_landscape_{utility_kind}.html",
        "q_policy": figure_dir / f"q_policy_{label}.html",
        "export_counts": figure_dir / f"export_counts_{label}.html",
        "start_counts": figure_dir / f"start_counts_{label}.html",
        "steps_by_start": figure_dir / f"steps_by_start_{label}.html",
        "start_to_export": figure_dir / f"start_to_export_{label}.html",
        "example_trajectories": figure_dir / f"example_trajectories_{label}.html",
    }

    xs = np.arange(G_MIN, G_MAX + 1)
    utility_vals = np.asarray(utility_fn(xs), dtype=float)

    if not paths["utility_landscape"].exists():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=utility_vals,
                mode="lines",
                name="Utility",
                line=dict(color="black", width=2),
            )
        )
        fig.update_layout(
            title=f"{get_utility_label(utility_kind)} landscape",
            xaxis_title="Poster state",
            yaxis_title="Utility",
            template="plotly_white",
        )
        fig.write_html(str(paths["utility_landscape"]), include_plotlyjs="cdn")

    q_export = user.Q[EXPORT]
    q_inc = user.Q[INC].copy()
    q_dec = user.Q[DEC].copy()
    q_inc[np.isneginf(q_inc)] = np.nan
    q_dec[np.isneginf(q_dec)] = np.nan

    p_export = np.zeros_like(xs, dtype=float)
    p_inc = np.zeros_like(xs, dtype=float)
    p_dec = np.zeros_like(xs, dtype=float)
    for k, g in enumerate(xs):
        probs = _policy_probs(user, int(g))
        p_export[k] = probs.get(EXPORT, 0.0)
        p_inc[k] = probs.get(INC, 0.0)
        p_dec[k] = probs.get(DEC, 0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Q* values", "Boltzmann action probabilities"),
    )
    fig.add_trace(go.Scatter(x=xs, y=q_export, mode="lines", name="Q export"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=q_inc, mode="lines", name="Q inc"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=q_dec, mode="lines", name="Q dec"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=p_export, mode="lines", name="P export"), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=p_inc, mode="lines", name="P inc"), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=p_dec, mode="lines", name="P dec"), row=2, col=1)
    fig.update_layout(title=f"{label}: Q* and policy", template="plotly_white", height=700)
    fig.update_xaxes(title_text="Poster state", row=2, col=1)
    fig.update_yaxes(title_text="Q value", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1, range=[-0.02, 1.02])
    fig.write_html(str(paths["q_policy"]), include_plotlyjs="cdn")

    export_counts = np.zeros(len(xs), dtype=int)
    for row in per_traj_rows:
        export_counts[int(row["export_poster"]) - G_MIN] += 1
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=xs, y=export_counts, name="Export count", marker_color="#3b82f6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=xs, y=utility_vals, mode="lines", name="Utility", line=dict(color="black")),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{label}: export-count distribution",
        xaxis_title="Poster state",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Export count", secondary_y=False)
    fig.update_yaxes(title_text="Utility", secondary_y=True)
    fig.write_html(str(paths["export_counts"]), include_plotlyjs="cdn")

    start_count_vals = np.array([start_counts[int(g)] for g in xs], dtype=int)
    fig = go.Figure(
        go.Bar(x=xs, y=start_count_vals, name="Start count", marker_color="#16a34a")
    )
    fig.update_layout(
        title=f"{label}: start-count schedule",
        xaxis_title="Poster state",
        yaxis_title="Start count",
        template="plotly_white",
    )
    fig.write_html(str(paths["start_counts"]), include_plotlyjs="cdn")

    start_x = np.array([r["start_poster"] for r in per_start_rows], dtype=int)
    steps_mean = np.array([r["steps_mean"] for r in per_start_rows], dtype=float)
    steps_std = np.array([r["steps_std"] for r in per_start_rows], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=start_x, y=steps_mean, mode="lines", name="Mean rows"))
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([start_x, start_x[::-1]]),
            y=np.concatenate([steps_mean + steps_std, (steps_mean - steps_std)[::-1]]),
            fill="toself",
            line=dict(color="rgba(59,130,246,0)"),
            fillcolor="rgba(59,130,246,0.18)",
            name="+/- 1 std",
        )
    )
    fig.update_layout(
        title=f"{label}: trajectory rows by start",
        xaxis_title="Start poster",
        yaxis_title="Rows per trajectory",
        template="plotly_white",
    )
    fig.write_html(str(paths["steps_by_start"]), include_plotlyjs="cdn")

    export_mean = np.array([r["export_poster_mean"] for r in per_start_rows], dtype=float)
    export_std = np.array([r["export_poster_std"] for r in per_start_rows], dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=start_x, y=export_mean, mode="lines", name="Mean exported poster")
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([start_x, start_x[::-1]]),
            y=np.concatenate([export_mean + export_std, (export_mean - export_std)[::-1]]),
            fill="toself",
            line=dict(color="rgba(59,130,246,0)"),
            fillcolor="rgba(59,130,246,0.18)",
            name="+/- 1 std",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=start_x,
            y=start_x,
            mode="lines",
            name="Identity",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{label}: exported poster by start",
        xaxis_title="Start poster",
        yaxis_title="Exported poster",
        template="plotly_white",
    )
    fig.write_html(str(paths["start_to_export"]), include_plotlyjs="cdn")

    fig = make_subplots(rows=len(DEFAULT_EXAMPLE_STARTS), cols=1, shared_xaxes=False)
    row_idx = 1
    for start in DEFAULT_EXAMPLE_STARTS:
        match = next((tr for tr in trajectories if int(tr["start_poster"]) == int(start)), None)
        if match is None:
            continue
        posters = np.array([int(step["poster"]) for step in match["traj"]], dtype=int)
        t = np.arange(len(posters))
        fig.add_trace(
            go.Scatter(
                x=t,
                y=posters,
                mode="lines+markers",
                name=f"start {start}",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=f"s={start}", row=row_idx, col=1)
        row_idx += 1
    fig.update_layout(
        title=f"{label}: example trajectories",
        template="plotly_white",
        height=max(360, 175 * (row_idx - 1)),
    )
    fig.update_xaxes(title_text="Time step", row=max(1, row_idx - 1), col=1)
    fig.write_html(str(paths["example_trajectories"]), include_plotlyjs="cdn")

    return {key: str(path) for key, path in paths.items()}


def _plot_export_counts(
    *,
    utility_fn: Callable,
    per_traj_rows: list[dict],
    title: str,
    save_path: Path,
    plt,
) -> None:
    xs = np.arange(G_MIN, G_MAX + 1)
    counts = np.zeros(len(xs), dtype=int)
    for row in per_traj_rows:
        counts[int(row["export_poster"]) - G_MIN] += 1
    utility_vals = utility_fn(xs)

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(xs, counts, width=0.85, color="tab:blue", alpha=0.75, label="Export count")
    ax1.set_xlabel("Poster state")
    ax1.set_ylabel("Export count")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, utility_vals, color="black", linewidth=2, label="Utility")
    ax2.set_ylabel("Utility")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_start_counts(
    *,
    start_counts: dict[int, int],
    title: str,
    save_path: Path,
    plt,
) -> None:
    xs = np.arange(G_MIN, G_MAX + 1)
    counts = np.array([start_counts[int(x)] for x in xs], dtype=int)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(xs, counts, width=0.85, color="tab:green", alpha=0.75)
    ax.set_xlabel("Poster state")
    ax.set_ylabel("Start count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_manifest(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    peak_basin: set[int],
    dataset_entries: list[dict],
    diagnostics_rows: list[dict],
    export_count_rows: list[dict],
    start_count_rows: list[dict],
) -> None:
    summary_dir = out_dir / "summaries"
    diagnostics_path = out_dir / "candidate_export_diagnostics.csv"
    export_counts_path = out_dir / "candidate_export_counts.csv"
    start_counts_path = out_dir / "candidate_start_counts.csv"

    _save_csv(diagnostics_path, diagnostics_rows)
    _save_csv(export_counts_path, export_count_rows)
    _save_csv(start_counts_path, start_count_rows)

    manifest = {
        "suite": "main_1d_candidates",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "state_space": "1d",
        "utilities": list(args.utilities),
        "balanced_configs": [
            {"gamma": float(g), "alpha": float(a)} for g, a in args.balanced_configs
        ],
        "balanced_equivalent_reps_per_state": list(args.balanced_reps),
        "clipped_configs": [
            {"gamma": float(g), "alpha": float(a)} for g, a in args.clipped_configs
        ],
        "clipped_equivalent_reps_per_state": list(args.clipped_reps),
        "peak_basin": {
            "centers": list(args.peak_centers),
            "radius": int(args.peak_radius),
            "states": sorted(int(g) for g in peak_basin),
            "note": (
                "The peak-clipped backup schedule includes every state at least "
                "once, but caps starts in these intended peak-basin states."
            ),
        },
        "max_steps": int(args.max_steps),
        "base_seed": int(args.seed),
        "shuffle_trajectories": bool(args.shuffle_trajectories),
        "diagnostics_csv": str(diagnostics_path),
        "export_counts_csv": str(export_counts_path),
        "start_counts_csv": str(start_counts_path),
        "summary_dir": str(summary_dir),
        "datasets": dataset_entries,
    }
    with (out_dir / "candidate_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _print_compact_table(rows: list[dict]) -> None:
    headers = [
        "dataset",
        "n_traj",
        "n_rows",
        "mean_len",
        "imm_pct",
        "start_eq_pct",
        "uniq_exp",
        "entropy",
        "top5_pct",
        "imm_peak_share",
    ]
    table = []
    for row in rows:
        table.append(
            [
                str(row["dataset"]),
                str(row["n_traj"]),
                str(row["n_rows"]),
                f"{float(row['mean_rows_per_traj']):.2f}",
                f"{100.0 * float(row['frac_immediate_export']):.1f}",
                f"{100.0 * float(row['frac_start_eq_export']):.1f}",
                str(row["n_unique_export_states"]),
                f"{float(row['export_entropy_norm']):.3f}",
                f"{100.0 * float(row['export_top5_share']):.1f}",
                f"{100.0 * float(row['frac_immediate_peak_basin_among_immediate']):.1f}",
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 1D Model 4 main-dataset candidates and export diagnostics."
        )
    )
    parser.add_argument(
        "--out-dir",
        default="data/main_1d_candidates",
        help="Directory for candidate datasets, summaries, plots, and manifest.",
    )
    parser.add_argument(
        "--utilities",
        default="smooth,jagged",
        help="Comma-separated utilities to generate: smooth,jagged.",
    )
    parser.add_argument(
        "--balanced-configs",
        default=",".join(f"{g}:{a}" for g, a in DEFAULT_BALANCED_CONFIGS),
        help="Comma-separated gamma:alpha pairs for uniform-start candidates.",
    )
    parser.add_argument(
        "--balanced-reps",
        default=",".join(str(x) for x in DEFAULT_REPS),
        help="Comma-separated repetitions per state for uniform-start candidates.",
    )
    parser.add_argument(
        "--clipped-configs",
        default=",".join(f"{g}:{a}" for g, a in DEFAULT_CLIPPED_CONFIGS),
        help="Comma-separated gamma:alpha pairs for peak-clipped backup candidates.",
    )
    parser.add_argument(
        "--clipped-reps",
        default=",".join(str(x) for x in DEFAULT_REPS),
        help="Comma-separated equivalent repetitions per state for clipped candidates.",
    )
    parser.add_argument(
        "--peak-centers",
        default=",".join(str(x) for x in DEFAULT_PEAK_CENTERS),
        help="Comma-separated intended utility peak centers used for basin diagnostics.",
    )
    parser.add_argument(
        "--peak-radius",
        type=int,
        default=5,
        help="Radius around each peak center for immediate-export basin diagnostics.",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument(
        "--no-shuffle-trajectories",
        action="store_true",
        help="Keep trajectories grouped by start instead of shuffling trajectory ids.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate data and CSV diagnostics without Matplotlib plots.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.utilities = tuple(u.strip().lower() for u in args.utilities.split(",") if u.strip())
    unknown = sorted(set(args.utilities) - set(VALID_UTILITY_KINDS))
    if unknown:
        raise ValueError(f"Unknown utilities: {unknown}. Expected {VALID_UTILITY_KINDS}")
    args.balanced_configs = _parse_float_pairs(args.balanced_configs)
    args.clipped_configs = _parse_float_pairs(args.clipped_configs)
    args.balanced_reps = _parse_ints(args.balanced_reps)
    args.clipped_reps = _parse_ints(args.clipped_reps)
    args.peak_centers = _parse_ints(args.peak_centers)
    args.shuffle_trajectories = not args.no_shuffle_trajectories

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peak_basin = _peak_basin_states(args.peak_centers, args.peak_radius)

    dataset_entries: list[dict] = []
    diagnostics_rows: list[dict] = []
    export_count_rows: list[dict] = []
    start_count_rows: list[dict] = []

    jobs: list[tuple[str, str, float, float, int, dict[int, int]]] = []
    for utility_kind in args.utilities:
        for gamma, alpha in args.balanced_configs:
            for reps in args.balanced_reps:
                jobs.append(
                    (
                        "uniform",
                        utility_kind,
                        float(gamma),
                        float(alpha),
                        int(reps),
                        _balanced_start_counts(int(reps)),
                    )
                )
        for gamma, alpha in args.clipped_configs:
            for reps in args.clipped_reps:
                jobs.append(
                    (
                        "peakclip",
                        utility_kind,
                        float(gamma),
                        float(alpha),
                        int(reps),
                        _peak_clipped_start_counts(int(reps), peak_basin),
                    )
                )

    print(f"Generating {len(jobs)} candidate datasets under {out_dir} ...")
    for schedule_kind, utility_kind, gamma, alpha, reps, start_counts in jobs:
        result = _generate_dataset(
            utility_kind=utility_kind,
            gamma=gamma,
            alpha=alpha,
            equivalent_reps=reps,
            schedule_kind=schedule_kind,
            start_counts=start_counts,
            out_dir=out_dir,
            max_steps=args.max_steps,
            base_seed=args.seed,
            shuffle=args.shuffle_trajectories,
            peak_basin=peak_basin,
            make_plots=not args.no_plots,
        )
        diagnostics = result["diagnostics"]
        diagnostics_rows.append(diagnostics)
        export_count_rows.extend(result["export_counts"])
        start_count_rows.extend(result["start_counts"])
        dataset_entries.append(diagnostics)
        print(
            f"  wrote {result['label']}: "
            f"n={diagnostics['n_traj']}, rows={diagnostics['n_rows']}, "
            f"immediate={100.0 * diagnostics['frac_immediate_export']:.1f}%"
        )

    _write_manifest(
        out_dir=out_dir,
        args=args,
        peak_basin=peak_basin,
        dataset_entries=dataset_entries,
        diagnostics_rows=diagnostics_rows,
        export_count_rows=export_count_rows,
        start_count_rows=start_count_rows,
    )
    print()
    _print_compact_table(diagnostics_rows)
    print()
    print(f"Manifest: {out_dir / 'candidate_manifest.json'}")
    print(f"Diagnostics: {out_dir / 'candidate_export_diagnostics.csv'}")


if __name__ == "__main__":
    main()
