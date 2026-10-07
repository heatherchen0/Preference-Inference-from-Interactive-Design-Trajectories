import numpy as np

def simulate_trajectories_with_random_starts(problem, user, n, rng, g_min=1, g_max=100, max_steps=10_000):
    all_trajs = []

    for traj_id in range(n):
        poster = int(rng.integers(g_min, g_max + 1))
        traj = []
        t = 0

        while True:
            action = user.pick_action(poster)
            traj.append({"traj_id": traj_id, "t": t, "poster": int(poster), "action": action})

            poster, done = problem.step(poster, action)
            t += 1

            if done:
                break
            if t >= max_steps:
                tail = traj[-30:] if len(traj) > 30 else traj
                raise RuntimeError(
                    f"Trajectory {traj_id} exceeded max_steps={max_steps}. "
                    f"Last state poster={poster}. Last steps tail={tail}"
                )

        all_trajs.append(traj)

    return all_trajs


def simulate_trajectories_by_start(
    problem,
    user,
    starts,
    n_reps_per_start=1,
    max_steps=10_000,
):
    """
    Simulate trajectories from a systematic list of starting posters.

    Parameters
    ----------
    problem : Problem1D
        Environment with step(...) defined.
    user : user model
        Any object with pick_action(poster).
    starts : iterable of int
        Starting poster values to sweep over.
    n_reps_per_start : int
        Number of repeated trajectories per start state.
        For deterministic users, this can be 1.
        For stochastic users, this can be > 1.
    max_steps : int
        Safety cap to prevent infinite loops.

    Returns
    -------
    all_trajs : list of list of dict
        Each trajectory is a list of step records.
    """
    all_trajs = []
    traj_id = 0

    for start_poster in starts:
        start_poster = int(start_poster)

        for rep in range(n_reps_per_start):
            poster = start_poster
            traj = []
            t = 0

            while True:
                action = user.pick_action(poster)

                traj.append(
                    {
                        "traj_id": traj_id,
                        "start_poster": start_poster,
                        "rep": rep,
                        "t": t,
                        "poster": int(poster),
                        "action": action,
                    }
                )

                poster, done = problem.step(poster, action)
                t += 1

                if done:
                    break

                if t >= max_steps:
                    tail = traj[-30:] if len(traj) > 30 else traj
                    raise RuntimeError(
                        f"Trajectory traj_id={traj_id} start={start_poster} rep={rep} "
                        f"exceeded max_steps={max_steps}. "
                        f"Last state poster={poster}. Last steps tail={tail}"
                    )

            all_trajs.append(traj)
            traj_id += 1

    return all_trajs


def trajectory_summary(traj, utility_fn):
    """
    Summarize one trajectory into one row.
    """
    start_poster = int(traj[0]["start_poster"]) if "start_poster" in traj[0] else int(traj[0]["poster"])
    export_poster = int(traj[-1]["poster"])
    steps = len(traj)
    export_utility = float(utility_fn(np.array([export_poster]))[0])

    row = {
        "start_poster": start_poster,
        "export_poster": export_poster,
        "steps": steps,
        "export_utility": export_utility,
    }

    if "rep" in traj[0]:
        row["rep"] = int(traj[0]["rep"])
    if "traj_id" in traj[0]:
        row["traj_id"] = int(traj[0]["traj_id"])

    return row


def summarize_by_start(per_traj_rows):
    """
    Aggregate trajectory summaries by start_poster.

    Parameters
    ----------
    per_traj_rows : list of dict
        Output rows from trajectory_summary(...)

    Returns
    -------
    per_start_rows : list of dict
        One aggregated row per start poster.
    """
    grouped = {}

    for row in per_traj_rows:
        start = int(row["start_poster"])
        if start not in grouped:
            grouped[start] = []
        grouped[start].append(row)

    per_start_rows = []

    for start in sorted(grouped.keys()):
        rows = grouped[start]

        export_posters = np.array([r["export_poster"] for r in rows], dtype=float)
        steps = np.array([r["steps"] for r in rows], dtype=float)
        export_utils = np.array([r["export_utility"] for r in rows], dtype=float)

        summary = {
            "start_poster": start,
            "n_trajs": len(rows),

            "mean_export_poster": float(np.mean(export_posters)),
            "std_export_poster": float(np.std(export_posters)),

            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),

            "mean_export_utility": float(np.mean(export_utils)),
            "std_export_utility": float(np.std(export_utils)),
        }

        per_start_rows.append(summary)

    return per_start_rows


def summarize_exports(per_traj_rows):
    export_posters = np.array([r["export_poster"] for r in per_traj_rows], dtype=int)
    steps = np.array([r["steps"] for r in per_traj_rows], dtype=int)
    utils = np.array([r["export_utility"] for r in per_traj_rows], dtype=float)

    print("  n =", len(per_traj_rows))
    print("  export poster: mean =", float(np.mean(export_posters)), "std =", float(np.std(export_posters)))
    print("  steps: mean =", float(np.mean(steps)), "std =", float(np.std(steps)))
    print("  export utility: mean =", float(np.mean(utils)), "std =", float(np.std(utils)))

    vals, counts = np.unique(export_posters, return_counts=True)
    order = np.argsort(-counts)
    topk = min(10, len(vals))
    print("  top export posters (poster: count):")
    for i in range(topk):
        v = int(vals[order[i]])
        c = int(counts[order[i]])
        print(f"    {v}: {c}")