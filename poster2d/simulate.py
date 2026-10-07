import numpy as np

from .env import DEFAULT_K, EXPORT, Problem2D


def _utility_at_poster(utility_fn, poster):
    value = np.asarray(utility_fn(np.asarray([int(poster)], dtype=int)), dtype=float)
    return float(value.reshape(-1)[0])


def _row_for_state(
    problem,
    utility_fn,
    *,
    t,
    poster,
    action,
    start_poster,
    rep,
):
    h, b = problem.from_poster(poster)
    start_h, start_b = problem.from_poster(start_poster)
    return {
        "t": int(t),
        "poster": int(poster),
        "headline_gray": int(h),
        "background_gray": int(b),
        "action": action,
        "start_poster": int(start_poster),
        "start_headline_gray": int(start_h),
        "start_background_gray": int(start_b),
        "rep": int(rep),
        "utility": _utility_at_poster(utility_fn, poster),
    }


def simulate_trajectory_from_start(
    problem,
    user,
    utility_fn,
    start_poster,
    rep=0,
    max_steps=10_000,
):
    poster = int(start_poster)
    problem.from_poster(poster)

    traj = []
    t = 0

    while True:
        action = user.pick_action(poster)
        traj.append(
            _row_for_state(
                problem,
                utility_fn,
                t=t,
                poster=poster,
                action=action,
                start_poster=start_poster,
                rep=rep,
            )
        )

        poster, done = problem.step(poster, action)
        t += 1

        if done:
            break
        if t >= max_steps:
            tail = traj[-30:] if len(traj) > 30 else traj
            raise RuntimeError(
                f"Trajectory from start={start_poster} rep={rep} "
                f"exceeded max_steps={max_steps}. Last poster={poster}. Tail={tail}"
            )

    if traj[-1]["action"] != EXPORT:
        raise RuntimeError("Trajectory terminated without an export action.")

    return traj


def summarize_trajectory(problem, traj):
    first = traj[0]
    last = traj[-1]
    start_poster = int(first["start_poster"])
    export_poster = int(last["poster"])
    start_h, start_b = problem.from_poster(start_poster)
    export_h, export_b = problem.from_poster(export_poster)

    return {
        "start_poster": start_poster,
        "start_headline_gray": int(start_h),
        "start_background_gray": int(start_b),
        "export_poster": export_poster,
        "export_headline_gray": int(export_h),
        "export_background_gray": int(export_b),
        "rep": int(first.get("rep", 0)),
        "steps": int(len(traj)),
        "edit_steps": int(max(0, len(traj) - 1)),
        "export_utility": float(last["utility"]),
    }


def run_systematic_sweep_2d(
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
        starts = problem.all_posters()
    starts = [int(s) for s in starts]

    if rng is None:
        rng = np.random.default_rng()

    trajectories = []
    for start in starts:
        for rep in range(int(n_reps_per_start)):
            traj = simulate_trajectory_from_start(
                problem=problem,
                user=user,
                utility_fn=utility_fn,
                start_poster=start,
                rep=rep,
                max_steps=max_steps,
            )
            trajectories.append(
                {
                    "start_poster": int(start),
                    "rep": int(rep),
                    "traj": traj,
                    "summary": summarize_trajectory(problem, traj),
                }
            )

    if shuffle_trajectories:
        rng.shuffle(trajectories)

    per_start_rows = summarize_by_start_2d(problem, [t["summary"] for t in trajectories])
    return {
        "trajectories": trajectories,
        "per_traj_rows": [dict(t["summary"]) for t in trajectories],
        "per_start_rows": per_start_rows,
    }


def materialize_step_records(
    trajectories,
    *,
    model_label,
    config_fields=None,
):
    config_fields = {} if config_fields is None else dict(config_fields)
    step_records = []
    per_traj_rows = []

    for traj_id, item in enumerate(trajectories):
        summary = dict(item["summary"])
        summary["traj_id"] = int(traj_id)
        summary["model"] = model_label
        summary.update(config_fields)
        per_traj_rows.append(summary)

        for row in item["traj"]:
            out = {
                "traj_id": int(traj_id),
                **row,
                "model": model_label,
            }
            out.update(config_fields)
            step_records.append(out)

    return step_records, per_traj_rows


def summarize_by_start_2d(problem, per_traj_rows):
    grouped = {}
    for row in per_traj_rows:
        grouped.setdefault(int(row["start_poster"]), []).append(row)

    summaries = []
    for start_poster in sorted(grouped):
        rows = grouped[start_poster]
        start_h, start_b = problem.from_poster(start_poster)
        export_posters = np.asarray([r["export_poster"] for r in rows], dtype=float)
        export_h = np.asarray([r["export_headline_gray"] for r in rows], dtype=float)
        export_b = np.asarray([r["export_background_gray"] for r in rows], dtype=float)
        steps = np.asarray([r["steps"] for r in rows], dtype=float)
        export_utils = np.asarray([r["export_utility"] for r in rows], dtype=float)
        summaries.append(
            {
                "start_poster": int(start_poster),
                "start_headline_gray": int(start_h),
                "start_background_gray": int(start_b),
                "n_trajs": int(len(rows)),
                "mean_export_poster": float(np.mean(export_posters)),
                "std_export_poster": float(np.std(export_posters)),
                "mean_export_headline_gray": float(np.mean(export_h)),
                "std_export_headline_gray": float(np.std(export_h)),
                "mean_export_background_gray": float(np.mean(export_b)),
                "std_export_background_gray": float(np.std(export_b)),
                "mean_steps": float(np.mean(steps)),
                "std_steps": float(np.std(steps)),
                "mean_export_utility": float(np.mean(export_utils)),
                "std_export_utility": float(np.std(export_utils)),
            }
        )
    return summaries


def build_problem_and_starts(k=DEFAULT_K):
    problem = Problem2D(k=k)
    return problem, problem.all_posters()
