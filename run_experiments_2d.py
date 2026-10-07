import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO

import numpy as np

from poster2d import (
    Problem2D,
    UserModel1GoalDirected2D,
    UserModel2Discounted2D,
    UserModel3Lookahead2D,
    UserModel4Boltzmann2D,
    smooth_contrast_interaction_default_params,
    smooth_contrast_interaction_grid,
)
from poster2d.plotting import (
    plot_example_trajectories_on_utility,
    plot_export_counts_heatmap,
    plot_model4_q_policy_heatmaps,
    plot_steps_by_start_heatmap,
    plot_utility_heatmap,
)
from poster2d.simulate import (
    materialize_step_records,
    run_systematic_sweep_2d,
    summarize_by_start_2d,
)


GAMMAS = (0.95,)
ALPHAS = (5,)
MODEL4_REPS_PER_START = 3
LOOKAHEAD_L = 2
EXAMPLE_START_STATES = ((1, 1), (1, 10), (5, 5), (6, 6), (10, 1), (10, 10)) #for plotting example trajectories
UTILITY_CHOICES = ("smooth_contrast_interaction",)


def _safe_label(label):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(label)).strip("_")


def _save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _matplotlib_available():
    try:
        stderr = StringIO()
        with redirect_stderr(stderr):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            plt.close("all")
        return True, ""
    except (AttributeError, ImportError) as exc:
        return False, str(exc)


def _make_run_root(output_root, run_root):
    if run_root:
        root = Path(run_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(output_root) / f"run_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "summaries").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    return root


def _utility_fn_from_grid(problem, utility_grid):
    values = np.asarray(utility_grid, dtype=float).reshape(-1)

    def utility_fn(posters):
        poster_arr = np.asarray(posters)
        scalar_input = poster_arr.ndim == 0
        flat_posters = poster_arr.reshape(-1)

        out = []
        for poster in flat_posters:
            out.append(float(values[problem.state_index(poster)]))

        out = np.asarray(out, dtype=float).reshape(poster_arr.shape)
        if scalar_input:
            return float(out)
        return out

    return utility_fn


def _cardinal_neighbor_values(U_grid, h, b):
    values = []
    for dh, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        hh = int(h) + dh
        bb = int(b) + db
        if 1 <= hh <= U_grid.shape[0] and 1 <= bb <= U_grid.shape[1]:
            values.append(float(U_grid[hh - 1, bb - 1]))
    return values


def _smooth_utility_landscape_summary(utility_grid):
    U = np.asarray(utility_grid, dtype=float)
    best_h_idx, best_b_idx = np.unravel_index(int(np.argmax(U)), U.shape)
    global_peak = (int(best_h_idx + 1), int(best_b_idx + 1))
    local_peak = (4, 9)
    bump = (4, 3)
    global_shoulder = _cardinal_neighbor_values(U, *global_peak)
    local_shoulder = _cardinal_neighbor_values(U, *local_peak)
    return {
        "global_peak": {
            "state": list(global_peak),
            "value": float(U[global_peak[0] - 1, global_peak[1] - 1]),
            "cardinal_shoulder_mean": float(np.mean(global_shoulder)),
            "cardinal_shoulder_values": global_shoulder,
        },
        "local_peak": {
            "state": list(local_peak),
            "value": float(U[local_peak[0] - 1, local_peak[1] - 1]),
            "cardinal_shoulder_mean": float(np.mean(local_shoulder)),
            "cardinal_shoulder_values": local_shoulder,
        },
        "bump": {
            "state": list(bump),
            "value": float(U[bump[0] - 1, bump[1] - 1]),
        },
    }


def _build_utility(name, problem):
    name = str(name).strip().lower()
    if name == "smooth_contrast_interaction":
        utility_grid = smooth_contrast_interaction_grid(k=problem.k)
        return {
            "name": name,
            "label": "Chosen 2D smooth contrast-interaction utility",
            "figure_name": "utility_smooth_contrast.png",
            "description": (
                "Chosen asymmetric smooth utility: dominant global peak at (8,3), "
                "weaker local peak near (4,9), and small bump near (4,3)."
            ),
            "params": smooth_contrast_interaction_default_params(k=problem.k),
            "landscape": _smooth_utility_landscape_summary(utility_grid),
            "grid": utility_grid,
            "fn": _utility_fn_from_grid(problem, utility_grid),
        }
    raise ValueError(f"Unknown utility {name!r}. Expected one of {UTILITY_CHOICES}.")


def _dataset_specs(utility_fn, k, seed, *, dry_run=False):
    specs = []

    def rng(offset):
        return np.random.default_rng(int(seed) + int(offset))

    specs.append(
        {
            "label": "UserModel1GoalDirected2D_train",
            "user": UserModel1GoalDirected2D(utility_fn, k=k, eps=0.0, rng=rng(101)),
            "n_reps": 1,
            "config": {"model_family": "model1_goal_directed", "eps": 0.0},
            "is_model4": False,
        }
    )

    gammas = (0.95,) if dry_run else GAMMAS
    alphas = (5,) if dry_run else ALPHAS

    for gamma in gammas:
        specs.append(
            {
                "label": f"UserModel2Discounted2D_gamma{gamma}_train",
                "user": UserModel2Discounted2D(
                    utility_fn,
                    k=k,
                    gamma=gamma,
                    eps=0.0,
                    rng=rng(200 + int(round(gamma * 1000))),
                ),
                "n_reps": 1,
                "config": {
                    "model_family": "model2_discounted",
                    "gamma": float(gamma),
                    "eps": 0.0,
                },
                "is_model4": False,
            }
        )

    specs.append(
        {
            "label": f"UserModel3Lookahead2D_L{LOOKAHEAD_L}_train",
            "user": UserModel3Lookahead2D(
                utility_fn,
                k=k,
                lookahead_k=LOOKAHEAD_L,
                eps=0.0,
                rng=rng(301),
            ),
            "n_reps": 1,
            "config": {
                "model_family": "model3_lookahead",
                "lookahead_L": int(LOOKAHEAD_L),
                "eps": 0.0,
            },
            "is_model4": False,
        }
    )

    for gamma in gammas:
        for alpha in alphas:
            specs.append(
                {
                    "label": f"UserModel4Boltzmann2D_gamma{gamma}_alpha{alpha}_train",
                    "user": UserModel4Boltzmann2D(
                        utility_fn,
                        k=k,
                        gamma=gamma,
                        alpha=alpha,
                        rng=rng(4000 + int(round(gamma * 1000)) + int(alpha)),
                    ),
                    "n_reps": 2 if dry_run else MODEL4_REPS_PER_START,
                    "config": {
                        "model_family": "model4_boltzmann",
                        "gamma": float(gamma),
                        "alpha": float(alpha),
                    },
                    "is_model4": True,
                }
            )

    return specs


def _dataset_summary(label, config, step_records, per_traj_rows, per_start_rows):
    export_posters = np.asarray([r["export_poster"] for r in per_traj_rows], dtype=int)
    steps = np.asarray([r["steps"] for r in per_traj_rows], dtype=float)
    export_utils = np.asarray([r["export_utility"] for r in per_traj_rows], dtype=float)
    unique_exports, export_counts = np.unique(export_posters, return_counts=True)
    order = np.argsort(-export_counts)

    return {
        "label": label,
        "config": dict(config),
        "n_rows": int(len(step_records)),
        "n_trajectories": int(len(per_traj_rows)),
        "n_start_states": int(len(per_start_rows)),
        "mean_steps": float(np.mean(steps)) if len(steps) else float("nan"),
        "std_steps": float(np.std(steps)) if len(steps) else float("nan"),
        "mean_export_utility": float(np.mean(export_utils)) if len(export_utils) else float("nan"),
        "std_export_utility": float(np.std(export_utils)) if len(export_utils) else float("nan"),
        "top_export_posters": [
            {
                "poster": int(unique_exports[i]),
                "count": int(export_counts[i]),
            }
            for i in order[:10]
        ],
    }


def _plot_dataset_diagnostics(
    *,
    problem,
    utility_grid,
    spec,
    sweep,
    per_traj_rows,
    per_start_rows,
    figures_dir,
    show,
):
    label_safe = _safe_label(spec["label"])
    starts_to_plot = [
        state for state in EXAMPLE_START_STATES if state[0] <= problem.k and state[1] <= problem.k
    ]

    plot_example_trajectories_on_utility(
        utility_grid,
        problem,
        sweep["trajectories"],
        starts_to_plot=starts_to_plot,
        title=f"{spec['label']}: example trajectories",
        save_path=figures_dir / f"trajectory_examples_{label_safe}.png",
        show=show,
    )
    plot_export_counts_heatmap(
        problem,
        per_traj_rows,
        title=f"{spec['label']}: export counts",
        save_path=figures_dir / f"export_counts_{label_safe}.png",
        show=show,
    )
    plot_steps_by_start_heatmap(
        problem,
        per_start_rows,
        title=f"{spec['label']}: mean steps by start",
        save_path=figures_dir / f"steps_by_start_{label_safe}.png",
        show=show,
    )

    if spec["is_model4"]:
        plot_model4_q_policy_heatmaps(
            spec["user"],
            title=f"{spec['label']}: Q* and policy",
            save_path=figures_dir / f"q_policy_{label_safe}.png",
            show=show,
        )


def run(args):
    run_root = _make_run_root(args.output_root, args.run_root)
    data_dir = run_root / "data"
    summaries_dir = run_root / "summaries"
    figures_dir = run_root / "figures"

    problem = Problem2D(k=args.k)
    utility = _build_utility(args.utility, problem)
    utility_fn = utility["fn"]
    utility_grid = utility["grid"]

    if args.dry_run:
        starts = [
            problem.to_poster(1, 1),
            problem.to_poster(1, problem.k),
            problem.to_poster(max(1, problem.k // 2), max(1, problem.k // 2)),
            problem.to_poster(problem.k, 1),
            problem.to_poster(problem.k, problem.k),
        ]
    else:
        starts = problem.all_posters()

    can_plot = False
    plot_skip_reason = ""
    if not args.no_plots:
        can_plot, plot_skip_reason = _matplotlib_available()
        if not can_plot:
            print(f"Plots skipped: {plot_skip_reason}")

    run_config = {
        "k": int(problem.k),
        "utility": utility["name"],
        "utility_label": utility["label"],
        "utility_preset": utility.get("preset", ""),
        "utility_description": utility.get("description", ""),
        "utility_params": utility.get("params", {}),
        "utility_landscape": utility.get("landscape", {}),
        "utility_revert_to": utility.get("revert_to", ""),
        "starts": starts,
        "dry_run": bool(args.dry_run),
        "shuffle_trajectories": True,
        "seed": int(args.seed),
        "gammas": list(GAMMAS),
        "alphas": list(ALPHAS),
        "model4_reps_per_start": int(MODEL4_REPS_PER_START),
        "lookahead_L": int(LOOKAHEAD_L),
        "plots_enabled": bool(can_plot),
        "plot_skip_reason": plot_skip_reason,
    }
    _save_json(run_root / "run_config.json", run_config)

    if can_plot:
        plot_utility_heatmap(
            utility_grid,
            title=utility["label"],
            save_path=figures_dir / utility["figure_name"],
            show=args.show_plots,
        )

    suite_summary = []
    for i, spec in enumerate(_dataset_specs(utility_fn, args.k, args.seed, dry_run=args.dry_run)):
        label = spec["label"]
        print(f"Generating {label} ...")
        sweep = run_systematic_sweep_2d(
            problem=problem,
            user=spec["user"],
            utility_fn=utility_fn,
            starts=starts,
            n_reps_per_start=spec["n_reps"],
            max_steps=args.max_steps,
            shuffle_trajectories=True,
            rng=np.random.default_rng(args.seed + 10_000 + i),
        )
        step_records, per_traj_rows = materialize_step_records(
            sweep["trajectories"],
            model_label=label,
            config_fields=spec["config"],
        )
        per_start_rows = summarize_by_start_2d(problem, per_traj_rows)

        label_safe = _safe_label(label)
        jsonl_path = data_dir / f"steps_{label_safe}.jsonl"
        per_traj_path = summaries_dir / f"summary_per_traj_{label_safe}.csv"
        per_start_path = summaries_dir / f"summary_per_start_{label_safe}.csv"
        summary_path = summaries_dir / f"summary_{label_safe}.json"

        _save_jsonl(jsonl_path, step_records)
        _save_csv(per_traj_path, per_traj_rows)
        _save_csv(per_start_path, per_start_rows)

        summary = _dataset_summary(
            label,
            spec["config"],
            step_records,
            per_traj_rows,
            per_start_rows,
        )
        summary["paths"] = {
            "steps_jsonl": str(jsonl_path),
            "per_traj_csv": str(per_traj_path),
            "per_start_csv": str(per_start_path),
        }
        _save_json(summary_path, summary)
        suite_summary.append(summary)

        if can_plot:
            _plot_dataset_diagnostics(
                problem=problem,
                utility_grid=utility_grid,
                spec=spec,
                sweep=sweep,
                per_traj_rows=per_traj_rows,
                per_start_rows=per_start_rows,
                figures_dir=figures_dir,
                show=args.show_plots,
            )

        print(f"  wrote {jsonl_path} ({len(per_traj_rows)} trajectories, {len(step_records)} rows)")

    _save_json(run_root / "suite_summary.json", suite_summary)
    print(f"Done. Outputs written under: {run_root}")
    return run_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2D poster trajectory datasets and diagnostics."
    )
    parser.add_argument("--output-root", default="outputs_2d")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--utility", choices=UTILITY_CHOICES, default="smooth_contrast_interaction")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
