from pathlib import Path

import numpy as np

from .env import (
    DEC_BACKGROUND,
    DEC_HEADLINE,
    EXPORT,
    INC_BACKGROUND,
    INC_HEADLINE,
    Problem2D,
)
from .utility import (
    MAIN_EFFECT_WEIGHT,
    PREFERRED_CONTRAST,
    smooth_contrast_interaction_grid,
)


MODEL4_ACTION_ORDER = (
    EXPORT,
    INC_HEADLINE,
    DEC_HEADLINE,
    INC_BACKGROUND,
    DEC_BACKGROUND,
)


def _finish_figure(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


def _coerce_start_poster(problem, start):
    if isinstance(start, tuple) and len(start) == 2:
        return problem.to_poster(start[0], start[1])
    return int(start)


def plot_utility_heatmap(
    U_grid,
    title="Smooth contrast-interaction utility",
    save_path=None,
    show=True,
    ax=None,
):
    import matplotlib.pyplot as plt

    U_grid = np.asarray(U_grid, dtype=float)
    if U_grid.ndim != 2:
        raise ValueError(f"U_grid must be 2D, got shape {U_grid.shape}.")
    if U_grid.shape[0] != U_grid.shape[1]:
        raise ValueError(f"U_grid must be square, got shape {U_grid.shape}.")
    if np.any(~np.isfinite(U_grid)):
        raise ValueError("U_grid must contain only finite values.")

    k = int(U_grid.shape[0])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    image = ax.imshow(
        U_grid,
        origin="lower",
        extent=(0.5, k + 0.5, 0.5, k + 0.5),
        aspect="equal",
        cmap="viridis",
    )

    best_h_idx, best_b_idx = np.unravel_index(int(np.argmax(U_grid)), U_grid.shape)
    best_h = best_h_idx + 1
    best_b = best_b_idx + 1
    ax.scatter(
        [best_b],
        [best_h],
        marker="x",
        s=90,
        linewidths=2.0,
        color="white",
        label=f"max ({best_h},{best_b})",
    )

    ax.set_title(title)
    ax.set_xlabel("Background grayscale b")
    ax.set_ylabel("Headline grayscale h")
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_yticks(np.arange(1, k + 1))
    ax.legend(loc="upper right", framealpha=0.85)
    fig.colorbar(image, ax=ax, label="Utility")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_example_trajectories_on_utility(
    U_grid,
    problem,
    trajectories,
    starts_to_plot,
    title="Example 2D trajectories",
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_utility_heatmap(U_grid, title=title, show=False, ax=ax)

    selected = []
    for start in starts_to_plot:
        start_poster = _coerce_start_poster(problem, start)
        match = next(
            (item for item in trajectories if int(item["start_poster"]) == start_poster),
            None,
        )
        if match is not None:
            selected.append(match)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(selected))))
    for item, color in zip(selected, colors):
        hs = [int(row["headline_gray"]) for row in item["traj"]]
        bs = [int(row["background_gray"]) for row in item["traj"]]
        start_h = int(item["traj"][0]["headline_gray"])
        start_b = int(item["traj"][0]["background_gray"])
        export_h = int(item["traj"][-1]["headline_gray"])
        export_b = int(item["traj"][-1]["background_gray"])

        ax.plot(bs, hs, marker="o", linewidth=1.8, markersize=4, color=color)
        ax.scatter([start_b], [start_h], marker="s", s=55, color=color)
        ax.scatter([export_b], [export_h], marker="*", s=110, color=color)

    _finish_figure(fig, save_path=save_path, show=show)
    if not show:
        plt.close(fig)
    return fig, ax


def plot_export_counts_heatmap(
    problem,
    per_traj_rows,
    title="Export-state counts",
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    counts = np.zeros((problem.k, problem.k), dtype=float)
    for row in per_traj_rows:
        h = int(row["export_headline_gray"])
        b = int(row["export_background_gray"])
        counts[h - 1, b - 1] += 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(
        counts,
        origin="lower",
        extent=(0.5, problem.k + 0.5, 0.5, problem.k + 0.5),
        aspect="equal",
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Background grayscale b")
    ax.set_ylabel("Headline grayscale h")
    ax.set_xticks(np.arange(1, problem.k + 1))
    ax.set_yticks(np.arange(1, problem.k + 1))
    fig.colorbar(image, ax=ax, label="Export count")

    _finish_figure(fig, save_path=save_path, show=show)
    if not show:
        plt.close(fig)
    return fig, ax


def plot_steps_by_start_heatmap(
    problem,
    per_start_rows,
    title="Mean steps to export by start",
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    values = np.full((problem.k, problem.k), np.nan, dtype=float)
    for row in per_start_rows:
        h = int(row["start_headline_gray"])
        b = int(row["start_background_gray"])
        values[h - 1, b - 1] = float(row["mean_steps"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(
        values,
        origin="lower",
        extent=(0.5, problem.k + 0.5, 0.5, problem.k + 0.5),
        aspect="equal",
        cmap="plasma",
    )
    ax.set_title(title)
    ax.set_xlabel("Background grayscale b")
    ax.set_ylabel("Headline grayscale h")
    ax.set_xticks(np.arange(1, problem.k + 1))
    ax.set_yticks(np.arange(1, problem.k + 1))
    fig.colorbar(image, ax=ax, label="Mean rows per trajectory")

    _finish_figure(fig, save_path=save_path, show=show)
    if not show:
        plt.close(fig)
    return fig, ax


def plot_model4_q_policy_heatmaps(
    user,
    title="Model 4 Q* and policy",
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    problem = user.problem if hasattr(user, "problem") else Problem2D()
    fig, axes = plt.subplots(
        2,
        len(MODEL4_ACTION_ORDER),
        figsize=(3.2 * len(MODEL4_ACTION_ORDER), 6.2),
        squeeze=False,
    )

    for col, action in enumerate(MODEL4_ACTION_ORDER):
        q_grid = np.asarray(user.Q[action], dtype=float).reshape(problem.k, problem.k)
        q_grid = q_grid.copy()
        q_grid[np.isneginf(q_grid)] = np.nan

        ax = axes[0, col]
        image = ax.imshow(q_grid, origin="lower", aspect="equal", cmap="viridis")
        ax.set_title(f"Q {action}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        p_grid = np.full((problem.k, problem.k), np.nan, dtype=float)
        for poster in problem.all_posters():
            h, b = problem.from_poster(poster)
            probs = user.action_probs(poster)
            if action in probs:
                p_grid[h - 1, b - 1] = float(probs[action])

        ax = axes[1, col]
        image = ax.imshow(
            p_grid,
            origin="lower",
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"P {action}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    _finish_figure(fig, save_path=save_path, show=show)
    if not show:
        plt.close(fig)
    return fig, axes


def plot_smooth_contrast_interaction_utility(
    k=10,
    preferred_contrast=PREFERRED_CONTRAST,
    contrast_width=None,
    main_effect_weight=MAIN_EFFECT_WEIGHT,
    background_scale=0.5,
    title="Smooth contrast-interaction utility",
    save_path=None,
    show=True,
    ax=None,
):
    U_grid = smooth_contrast_interaction_grid(
        k=k,
        preferred_contrast=preferred_contrast,
        contrast_width=contrast_width,
        main_effect_weight=main_effect_weight,
        background_scale=background_scale,
    )
    return plot_utility_heatmap(
        U_grid,
        title=title,
        save_path=save_path,
        show=show,
        ax=ax,
    )
if __name__ == "__main__":
    plot_smooth_contrast_interaction_utility()
