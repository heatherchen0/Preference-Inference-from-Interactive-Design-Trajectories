import numpy as np
from poster_env import Problem1D, G_MIN, G_MAX

def run_trajectory_from_start(problem, user, start_poster, utility_fn, max_steps=10_000):
    poster = int(start_poster)
    traj = []
    t = 0

    while True:
        action = user.pick_action(poster)
        traj.append({
            "t": t,
            "poster": int(poster),
            "action": action,
        })

        poster, done = problem.step(poster, action)
        t += 1

        if done:
            break
        if t >= max_steps:
            tail = traj[-30:] if len(traj) > 30 else traj
            raise RuntimeError(
                f"Trajectory from start={start_poster} exceeded max_steps={max_steps}. "
                f"Last poster={poster}. Tail={tail}"
            )

    export_poster = int(traj[-1]["poster"])
    export_utility = float(utility_fn(np.array([export_poster]))[0])

    summary = {
        "start_poster": int(start_poster),
        "export_poster": export_poster,
        "steps": len(traj),
        "export_utility": export_utility,
    }
    return traj, summary

# def run_systematic_sweep(problem, user, utility_fn, starts=None, n_reps_per_start=1, max_steps=10_000):
#     if starts is None:
#         starts = np.arange(problem.g_min, problem.g_max + 1)

#     per_traj_rows = []
#     trajectories = []

#     for start in starts:
#         for rep in range(n_reps_per_start):
#             traj, summary = run_trajectory_from_start(
#                 problem=problem,
#                 user=user,
#                 start_poster=int(start),
#                 utility_fn=utility_fn,
#                 max_steps=max_steps,
#             )
#             summary["rep"] = rep
#             per_traj_rows.append(summary)

#             trajectories.append({
#                 "start_poster": int(start),
#                 "rep": rep,
#                 "traj": traj,
#             })

#     # aggregate by start
#     per_start_rows = []
#     for start in starts:
#         rows = [r for r in per_traj_rows if r["start_poster"] == int(start)]

#         export_posters = np.array([r["export_poster"] for r in rows], dtype=float)
#         steps = np.array([r["steps"] for r in rows], dtype=float)
#         export_utils = np.array([r["export_utility"] for r in rows], dtype=float)

#         per_start_rows.append({
#             "start_poster": int(start),
#             "n": len(rows),
#             "export_poster_mean": float(np.mean(export_posters)),
#             "export_poster_std": float(np.std(export_posters)),
#             "steps_mean": float(np.mean(steps)),
#             "steps_std": float(np.std(steps)),
#             "export_utility_mean": float(np.mean(export_utils)),
#             "export_utility_std": float(np.std(export_utils)),
#         })

#     return {
#         "per_traj_rows": per_traj_rows,
#         "per_start_rows": per_start_rows,
#         "trajectories": trajectories,
#     }

def run_systematic_sweep(
    problem,
    user,
    utility_fn,
    starts=None,
    n_reps_per_start=1,
    max_steps=10_000,
    shuffle_trajectories=True,
    rng=None,
):
    if starts is None:
        starts = np.arange(problem.g_min, problem.g_max + 1)

    if rng is None:
        rng = np.random.default_rng()

    trajectories = []

    for start in starts:
        start = int(start)
        for rep in range(int(n_reps_per_start)):
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
                    "start_poster": start,
                    "rep": int(rep),
                    "traj": traj,
                    "summary": summary,
                }
            )

    if shuffle_trajectories:
        rng.shuffle(trajectories)

    # Per-trajectory summaries (in the same order as `trajectories`)
    per_traj_rows = [item["summary"] for item in trajectories]

    # Aggregate by start (order-independent)
    starts_list = [int(s) for s in starts]
    per_start_rows = []
    for start in starts_list:
        rows = [r for r in per_traj_rows if r["start_poster"] == start]

        export_posters = np.array([r["export_poster"] for r in rows], dtype=float)
        steps = np.array([r["steps"] for r in rows], dtype=float)
        export_utils = np.array([r["export_utility"] for r in rows], dtype=float)

        per_start_rows.append(
            {
                "start_poster": start,
                "n": len(rows),
                "export_poster_mean": float(np.mean(export_posters)) if len(rows) else float("nan"),
                "export_poster_std": float(np.std(export_posters)) if len(rows) else float("nan"),
                "steps_mean": float(np.mean(steps)) if len(rows) else float("nan"),
                "steps_std": float(np.std(steps)) if len(rows) else float("nan"),
                "export_utility_mean": float(np.mean(export_utils)) if len(rows) else float("nan"),
                "export_utility_std": float(np.std(export_utils)) if len(rows) else float("nan"),
            }
        )

    return {
        "per_traj_rows": per_traj_rows,
        "per_start_rows": per_start_rows,
        "trajectories": trajectories,
    }