import numpy as np

from .state_space import (
    ACTION_MAP_1D,
    coerce_action,
    normalize_state_space,
    action_order,
    poster_to_coord_2d,
)

ACTION_MAP = ACTION_MAP_1D


def _coerce_action(a, *, state_space: str = "1d"):
    return coerce_action(a, state_space=state_space)


def prepare_inverse_data(data, poster_min=1, poster_max=100, *, state_space: str = "1d"):
    state_space = normalize_state_space(state_space)
    num_states = int(poster_max - poster_min + 1)

    if hasattr(data, "columns") and hasattr(data, "__len__"):
        if "poster" not in data.columns or "action" not in data.columns:
            raise ValueError("DataFrame must have columns ['poster','action']")
        posters = data["poster"].to_numpy()
        actions = data["action"].to_numpy()
    else:
        posters = []
        actions = []
        for r in data:
            p = None
            for k in ("poster", "poster_id", "posterIndex", "poster_idx"):
                if k in r:
                    p = r[k]
                    break
            if p is None:
                raise KeyError(f"Could not find poster id field in row keys: {list(r.keys())}")

            if "action" not in r:
                raise KeyError(f"Could not find action field in row keys: {list(r.keys())}")

            posters.append(p)
            actions.append(r["action"])

        posters = np.asarray(posters)
        actions = np.asarray(actions)

    posters = posters.astype(int)
    action_idx = np.asarray(
        [_coerce_action(a, state_space=state_space) for a in actions],
        dtype=int,
    )

    if np.any(posters < poster_min) or np.any(posters > poster_max):
        bad = posters[(posters < poster_min) | (posters > poster_max)][0]
        raise ValueError(f"poster {bad} out of range [{poster_min},{poster_max}]")

    n_actions = len(action_order(state_space))
    if np.any((action_idx < 0) | (action_idx >= n_actions)):
        bad = action_idx[(action_idx < 0) | (action_idx >= n_actions)][0]
        raise ValueError(f"action {bad} not in [0,{n_actions - 1}]")

    state_idx = posters - int(poster_min)
    state_idx = state_idx.astype(int)

    return state_idx, action_idx, num_states


def summarize_inverse_data(inv, *, state_space: str = "1d"):
    state_space = normalize_state_space(state_space)
    poster_ids = inv["poster_ids"]
    actions = inv["actions"]
    names = action_order(state_space)
    out = {
        "n_obs": int(inv["n_obs"]),
        "poster_min": int(inv["poster_min"]),
        "poster_max": int(inv["poster_max"]),
        "unique_posters": int(len(np.unique(poster_ids))),
        "action_counts": {
            names[i]: int(np.sum(actions == i))
            for i in range(len(names))
        },
    }
    return out


def collect_data_summary(file_path, gamma_fixed):
    return {
        "file": str(file_path),
        "gamma_fixed": gamma_fixed,
    }


def prepare_pbo_start_end_data(
    data,
    poster_min=1,
    poster_max=100,
    *,
    drop_ties: bool = True,
    state_space: str = "1d",
    grid_k: int | None = None,
):
    """
    Convert full trajectories into PBO start-end comparisons.

    Each trajectory contributes one comparison endpoint > start when
    endpoint != start. Ties are dropped by default because they contribute
    a constant sigmoid(0)=0.5 likelihood term.
    """
    num_states = int(poster_max - poster_min + 1)

    if hasattr(data, "columns") and hasattr(data, "__len__"):
        required = {"traj_id", "t", "poster"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"prepare_pbo_start_end_data: missing required columns: {missing}")
        records = data.sort_values(["traj_id", "t"]).to_dict("records")
    else:
        records = list(data)
        if records:
            required = {"traj_id", "t", "poster"}
            missing = required - set(records[0].keys())
            if missing:
                raise ValueError(f"prepare_pbo_start_end_data: missing required columns: {missing}")
        records = sorted(records, key=lambda r: (r["traj_id"], int(r["t"])))

    starts = []
    ends = []
    traj_ids = []

    current_tid = None
    current_start = None
    current_end = None

    for r in records:
        tid = r["traj_id"]
        poster = int(r["poster"])
        if current_tid is None or tid != current_tid:
            if current_tid is not None:
                traj_ids.append(current_tid)
                starts.append(int(current_start))
                ends.append(int(current_end))
            current_tid = tid
            current_start = poster
        current_end = poster

    if current_tid is not None:
        traj_ids.append(current_tid)
        starts.append(int(current_start))
        ends.append(int(current_end))

    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)

    if starts.size:
        all_posters = np.concatenate([starts, ends])
        if np.any(all_posters < poster_min) or np.any(all_posters > poster_max):
            bad = all_posters[(all_posters < poster_min) | (all_posters > poster_max)][0]
            raise ValueError(f"poster {bad} out of range [{poster_min},{poster_max}]")

    tie_mask = starts == ends
    if drop_ties:
        keep = ~tie_mask
    else:
        keep = np.ones_like(tie_mask, dtype=bool)

    start_kept = starts[keep]
    end_kept = ends[keep]

    start_idx = (start_kept - int(poster_min)).astype(int)
    end_idx = (end_kept - int(poster_min)).astype(int)

    stats = {
        "pbo_comparison_type": "start-end",
        "n_traj_total": int(len(starts)),
        "n_action_obs": int(len(records)),
        "n_pbo_duels": int(len(start_idx)),
        "n_ties_dropped": int(np.sum(tie_mask)) if drop_ties else 0,
        "n_start_end_ties": int(np.sum(tie_mask)),
        "n_transition_ties": 0,
        "n_transition_candidates": 0,
        "drop_ties": bool(drop_ties),
        "tie_fraction": float(np.mean(tie_mask)) if len(tie_mask) else float("nan"),
    }

    kept_traj_ids = np.asarray(traj_ids, dtype=object)[keep]
    comparisons = []
    for i in range(len(start_idx)):
        row = {
            "traj_id": kept_traj_ids[i],
            "start_poster": int(start_kept[i]),
            "end_poster": int(end_kept[i]),
            "start_idx": int(start_idx[i]),
            "end_idx": int(end_idx[i]),
        }
        if normalize_state_space(state_space) == "2d" and grid_k is not None:
            sh, sb = poster_to_coord_2d(row["start_poster"], grid_k=int(grid_k))
            eh, eb = poster_to_coord_2d(row["end_poster"], grid_k=int(grid_k))
            row.update(
                {
                    "start_headline_gray": int(sh),
                    "start_background_gray": int(sb),
                    "end_headline_gray": int(eh),
                    "end_background_gray": int(eb),
                }
            )
        comparisons.append(row)

    return start_idx, end_idx, num_states, stats, comparisons


def prepare_pbo_transition_data(
    data,
    poster_min=1,
    poster_max=100,
    *,
    drop_ties: bool = True,
    state_space: str = "1d",
    grid_k: int | None = None,
):
    """
    Convert full trajectories into adjacent-transition PBO comparisons.

    Each observed transition contributes one comparison next_state > current_state.
    Repeated and reversed transitions are kept as noisy evidence.
    """
    num_states = int(poster_max - poster_min + 1)

    if hasattr(data, "columns") and hasattr(data, "__len__"):
        required = {"traj_id", "t", "poster"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"prepare_pbo_transition_data: missing required columns: {missing}")
        records = data.sort_values(["traj_id", "t"]).to_dict("records")
    else:
        records = list(data)
        if records:
            required = {"traj_id", "t", "poster"}
            missing = required - set(records[0].keys())
            if missing:
                raise ValueError(f"prepare_pbo_transition_data: missing required columns: {missing}")
        records = sorted(records, key=lambda r: (r["traj_id"], int(r["t"])))

    starts = []
    ends = []
    traj_ids = []
    ts = []
    n_traj_total = 0

    current_tid = None
    current_rows = []

    def _flush_current():
        nonlocal n_traj_total
        if not current_rows:
            return
        n_traj_total += 1
        for a, b in zip(current_rows[:-1], current_rows[1:]):
            starts.append(int(a["poster"]))
            ends.append(int(b["poster"]))
            traj_ids.append(a["traj_id"])
            ts.append(int(a["t"]))

    for r in records:
        tid = r["traj_id"]
        if current_tid is None or tid != current_tid:
            _flush_current()
            current_tid = tid
            current_rows = []
        current_rows.append(r)

    _flush_current()

    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)

    if starts.size:
        all_posters = np.concatenate([starts, ends])
        if np.any(all_posters < poster_min) or np.any(all_posters > poster_max):
            bad = all_posters[(all_posters < poster_min) | (all_posters > poster_max)][0]
            raise ValueError(f"poster {bad} out of range [{poster_min},{poster_max}]")

    tie_mask = starts == ends
    if drop_ties:
        keep = ~tie_mask
    else:
        keep = np.ones_like(tie_mask, dtype=bool)

    start_kept = starts[keep]
    end_kept = ends[keep]

    start_idx = (start_kept - int(poster_min)).astype(int)
    end_idx = (end_kept - int(poster_min)).astype(int)

    n_transition_candidates = int(len(starts))
    n_transition_ties = int(np.sum(tie_mask))
    stats = {
        "pbo_comparison_type": "transition",
        "n_traj_total": int(n_traj_total),
        "n_action_obs": int(len(records)),
        "n_transition_candidates": n_transition_candidates,
        "n_pbo_duels": int(len(start_idx)),
        "n_ties_dropped": n_transition_ties if drop_ties else 0,
        "n_transition_ties": n_transition_ties,
        "n_start_end_ties": 0,
        "drop_ties": bool(drop_ties),
        "tie_fraction": (
            float(np.mean(tie_mask)) if n_transition_candidates else float("nan")
        ),
    }

    kept_traj_ids = np.asarray(traj_ids, dtype=object)[keep]
    kept_ts = np.asarray(ts, dtype=int)[keep]
    comparisons = []
    for i in range(len(start_idx)):
        row = {
            "traj_id": kept_traj_ids[i],
            "t": int(kept_ts[i]),
            "start_poster": int(start_kept[i]),
            "end_poster": int(end_kept[i]),
            "start_idx": int(start_idx[i]),
            "end_idx": int(end_idx[i]),
        }
        if normalize_state_space(state_space) == "2d" and grid_k is not None:
            sh, sb = poster_to_coord_2d(row["start_poster"], grid_k=int(grid_k))
            eh, eb = poster_to_coord_2d(row["end_poster"], grid_k=int(grid_k))
            row.update(
                {
                    "start_headline_gray": int(sh),
                    "start_background_gray": int(sb),
                    "end_headline_gray": int(eh),
                    "end_background_gray": int(eb),
                }
            )
        comparisons.append(row)

    return start_idx, end_idx, num_states, stats, comparisons
