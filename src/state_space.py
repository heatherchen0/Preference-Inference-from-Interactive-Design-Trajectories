from __future__ import annotations

import math
from typing import Iterable

import numpy as np


STATE_SPACE_1D = "1d"
STATE_SPACE_2D = "2d"

ACTION_ORDER_1D = ("dec", "export", "inc")
ACTION_ORDER_2D = (
    "export",
    "inc-headline",
    "dec-headline",
    "inc-background",
    "dec-background",
)

ACTION_MAP_1D = {
    "left": 0,
    "dec": 0,
    "stay": 1,
    "export": 1,
    "right": 2,
    "inc": 2,
}

ACTION_MAP_2D = {
    "export": 0,
    "inc-headline": 1,
    "dec-headline": 2,
    "inc-background": 3,
    "dec-background": 4,
}


def normalize_state_space(state_space: str | None) -> str:
    s = str(state_space or STATE_SPACE_1D).strip().lower()
    aliases = {
        "1": STATE_SPACE_1D,
        "one_d": STATE_SPACE_1D,
        "one-dimensional": STATE_SPACE_1D,
        "1d": STATE_SPACE_1D,
        "2": STATE_SPACE_2D,
        "two_d": STATE_SPACE_2D,
        "two-dimensional": STATE_SPACE_2D,
        "2d": STATE_SPACE_2D,
    }
    if s not in aliases:
        raise ValueError(f"Unknown state_space={state_space!r}; use '1d' or '2d'.")
    return aliases[s]


def infer_grid_k(num_states: int, grid_k: int | None = None) -> int:
    if grid_k is not None:
        k = int(grid_k)
        if k <= 0:
            raise ValueError(f"grid_k must be positive, got {grid_k!r}.")
        if k * k != int(num_states):
            raise ValueError(
                f"grid_k={k} implies {k*k} states, but num_states={num_states}."
            )
        return k

    root = int(round(math.sqrt(int(num_states))))
    if root * root != int(num_states):
        raise ValueError(
            f"Cannot infer square 2D grid size from num_states={num_states}."
        )
    return root


def action_order(state_space: str | None = STATE_SPACE_1D) -> tuple[str, ...]:
    return ACTION_ORDER_2D if normalize_state_space(state_space) == STATE_SPACE_2D else ACTION_ORDER_1D


def action_map(state_space: str | None = STATE_SPACE_1D) -> dict[str, int]:
    return ACTION_MAP_2D if normalize_state_space(state_space) == STATE_SPACE_2D else ACTION_MAP_1D


def coerce_action(action, *, state_space: str | None = STATE_SPACE_1D) -> int:
    if isinstance(action, str):
        a = action.strip().lower()
        mapping = action_map(state_space)
        if a not in mapping:
            raise ValueError(f"Unknown {normalize_state_space(state_space)} action string: {action}")
        return int(mapping[a])
    return int(action)


def state_index_to_coord_2d(index, *, grid_k: int) -> tuple[int, int]:
    k = int(grid_k)
    i = int(index)
    if i < 0 or i >= k * k:
        raise ValueError(f"state index {index!r} out of range [0,{k*k - 1}].")
    return i // k + 1, i % k + 1


def poster_to_coord_2d(poster, *, grid_k: int) -> tuple[int, int]:
    return state_index_to_coord_2d(int(poster) - 1, grid_k=int(grid_k))


def coord_to_poster_2d(h: int, b: int, *, grid_k: int) -> int:
    k = int(grid_k)
    h = int(h)
    b = int(b)
    if h < 1 or h > k or b < 1 or b > k:
        raise ValueError(f"(h,b)=({h},{b}) out of range [1,{k}] x [1,{k}].")
    return (h - 1) * k + b


def coordinate_arrays_2d(*, grid_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(grid_k)
    coords = np.arange(1, k + 1, dtype=np.int64)
    H, B = np.meshgrid(coords, coords, indexing="ij")
    return H.reshape(-1), B.reshape(-1)


def normalized_gp_inputs(
    *,
    state_space: str | None,
    num_states: int,
    grid_k: int | None = None,
) -> np.ndarray:
    space = normalize_state_space(state_space)
    S = int(num_states)
    if space == STATE_SPACE_1D:
        if S <= 1:
            raise ValueError("num_states must be >= 2 for GP input rescaling.")
        return np.linspace(0.0, 1.0, S, dtype=np.float64)[:, None]

    k = infer_grid_k(S, grid_k=grid_k)
    h, b = coordinate_arrays_2d(grid_k=k)
    denom = float(max(k - 1, 1))
    return np.column_stack(
        [
            (h.astype(np.float64) - 1.0) / denom,
            (b.astype(np.float64) - 1.0) / denom,
        ]
    )


def manhattan_distance_matrix_2d(*, grid_k: int) -> np.ndarray:
    h, b = coordinate_arrays_2d(grid_k=int(grid_k))
    return (
        np.abs(h[:, None] - h[None, :])
        + np.abs(b[:, None] - b[None, :])
    ).astype(np.float64)


def neighbor_indices_and_mask_2d(*, grid_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(grid_k)
    S = k * k
    neighbors = np.zeros((S, len(ACTION_ORDER_2D)), dtype=np.int64)
    valid = np.zeros((S, len(ACTION_ORDER_2D)), dtype=bool)

    for idx in range(S):
        h, b = state_index_to_coord_2d(idx, grid_k=k)
        neighbors[idx, 0] = idx
        valid[idx, 0] = True

        candidates = {
            1: (h + 1, b, h < k),
            2: (h - 1, b, h > 1),
            3: (h, b + 1, b < k),
            4: (h, b - 1, b > 1),
        }
        for action_idx, (hh, bb, ok) in candidates.items():
            valid[idx, action_idx] = bool(ok)
            neighbors[idx, action_idx] = (
                coord_to_poster_2d(hh, bb, grid_k=k) - 1 if ok else idx
            )

    return neighbors, valid


def contrast_levels_2d(*, grid_k: int) -> np.ndarray:
    h, b = coordinate_arrays_2d(grid_k=int(grid_k))
    return np.abs(h - b).astype(np.int64)


def contrast_profile(values: Iterable[float], *, grid_k: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    k = int(grid_k)
    if vals.shape[0] != k * k:
        raise ValueError(f"Expected {k*k} values for grid_k={k}, got {vals.shape[0]}.")

    c = contrast_levels_2d(grid_k=k)
    out = np.empty((k,), dtype=float)
    for level in range(k):
        out[level] = float(np.mean(vals[c == level]))
    return out
