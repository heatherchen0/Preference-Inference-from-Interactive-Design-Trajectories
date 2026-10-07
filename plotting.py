import numpy as np
import matplotlib.pyplot as plt
from poster_env import INC, DEC, EXPORT
from user_models import _softmax_stable, UserModel4_EndpointDiscountedBoltzmann
from utility import utility

def _save_show(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_utility_landscape(
    utility_fn,
    g_min,
    g_max,
    title="Utility landscape",
    save_path=None,
    show=True,
):
    xs = np.arange(g_min, g_max + 1)
    ys = utility_fn(xs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, linewidth=2, label="Utility")

    global_idx = int(np.argmax(ys))
    global_g = int(xs[global_idx])
    global_u = float(ys[global_idx])

    ax.scatter([global_g], [global_u], s=80, label=f"Global max at {global_g}", zorder=3)

    # local maxima on the discrete grid
    local_maxima = []
    for i in range(len(xs)):
        left_ok = (i == 0) or (ys[i] >= ys[i - 1])
        right_ok = (i == len(xs) - 1) or (ys[i] >= ys[i + 1])
        strict = ((i > 0 and ys[i] > ys[i - 1]) or (i < len(xs) - 1 and ys[i] > ys[i + 1]))
        if left_ok and right_ok and strict:
            local_maxima.append((int(xs[i]), float(ys[i])))

    if local_maxima:
        ax.scatter(
            [g for g, _ in local_maxima],
            [u for _, u in local_maxima],
            s=40,
            alpha=0.8,
            label="Local maxima",
            zorder=3,
        )

    ax.set_xlabel("Poster")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    _save_show(fig, save_path=save_path, show=show)

def plot_start_to_export(
    per_start_rows,
    title="Start to exported poster",
    save_path=None,
    show=True,
    with_band=False,
):
    x = np.array([r["start_poster"] for r in per_start_rows])
    y = np.array([r["export_poster_mean"] for r in per_start_rows])
    ystd = np.array([r["export_poster_std"] for r in per_start_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2, label="Mean exported poster")
    ax.plot(x, x, linestyle="--", alpha=0.5, label="Identity")

    if with_band and np.any(ystd > 0):
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")

    ax.set_xlabel("Start poster")
    ax.set_ylabel("Exported poster")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    _save_show(fig, save_path=save_path, show=show)

def plot_steps_vs_start(
    per_start_rows,
    title="Steps vs start",
    save_path=None,
    show=True,
    with_band=False,
):
    x = np.array([r["start_poster"] for r in per_start_rows])
    y = np.array([r["steps_mean"] for r in per_start_rows])
    ystd = np.array([r["steps_std"] for r in per_start_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2, label="Mean steps")
    if with_band and np.any(ystd > 0):
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")

    ax.set_xlabel("Start poster")
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    _save_show(fig, save_path=save_path, show=show)

def plot_export_utility_vs_start(
    per_start_rows,
    title="Export utility vs start",
    save_path=None,
    show=True,
    with_band=False,
):
    x = np.array([r["start_poster"] for r in per_start_rows])
    y = np.array([r["export_utility_mean"] for r in per_start_rows])
    ystd = np.array([r["export_utility_std"] for r in per_start_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2, label="Mean export utility")
    if with_band and np.any(ystd > 0):
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")

    ax.set_xlabel("Start poster")
    ax.set_ylabel("Export utility")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    _save_show(fig, save_path=save_path, show=show)

def plot_overlay_exports_on_utility(
    utility_fn,
    per_traj_rows,
    per_start_rows,
    g_min,
    g_max,
    title="Exported posters over utility landscape",
    save_path=None,
    show=True,
):
    xs = np.arange(g_min, g_max + 1)
    ys = utility_fn(xs)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(xs, ys, color="black", linewidth=2, label="Utility")
    ax1.set_xlabel("Poster")
    ax1.set_ylabel("Utility", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()

    starts = np.array([r["start_poster"] for r in per_traj_rows], dtype=float)
    exports = np.array([r["export_poster"] for r in per_traj_rows], dtype=float)
    ax2.scatter(starts, exports, s=10, alpha=0.12, color="tab:blue", label="Trajectories")

    x = np.array([r["start_poster"] for r in per_start_rows], dtype=float)
    y = np.array([r["export_poster_mean"] for r in per_start_rows], dtype=float)
    ax2.plot(x, y, color="tab:red", linewidth=2, label="Mean exported poster")
    ax2.set_ylabel("Exported poster", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(g_min, g_max)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title)

    _save_show(fig, save_path=save_path, show=show)

def plot_example_trajectories(
    utility_fn,
    trajectories,
    starts_to_plot,
    title="Example trajectories",
    save_path=None,
    show=True,
):
    xs = sorted(set(int(s) for s in starts_to_plot))
    selected = []

    for s in xs:
        matches = [tr for tr in trajectories if tr["start_poster"] == s]
        if len(matches) > 0:
            selected.append(matches[0])

    n = len(selected)
    if n == 0:
        raise ValueError("No matching trajectories found for starts_to_plot")

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    utility_x = np.arange(min(xs) - 5, max(xs) + 6)
    utility_x = utility_x[(utility_x >= 1) & (utility_x <= 100)]

    for ax, item in zip(axes, selected):
        traj = item["traj"]
        posters = [step["poster"] for step in traj]
        t = np.arange(len(posters))
        u = utility_fn(np.array(posters))

        ax2 = ax.twinx()
        ax.plot(t, posters, marker="o", linewidth=2, label="Poster path")
        ax2.plot(t, u, linestyle="--", alpha=0.7, color="tab:orange", label="Utility along path")

        ax.set_ylabel(f"Start {item['start_poster']}\nPoster")
        ax2.set_ylabel("Utility")
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    axes[-1].set_xlabel("Time step")
    fig.suptitle(title, y=1.02)

    _save_show(fig, save_path=save_path, show=show)

def plot_combined_metric(results_dict, metric_key, ylabel, title, save_path=None, show=True):
    labels = list(results_dict.keys())
    n = len(labels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, label in zip(axes, labels):
        per_start_rows = results_dict[label]["per_start_rows"]
        x = np.array([r["start_poster"] for r in per_start_rows])

        if metric_key == "export_poster":
            y = np.array([r["export_poster_mean"] for r in per_start_rows])
            ystd = np.array([r["export_poster_std"] for r in per_start_rows])
            ax.plot(x, y, linewidth=2, label="Mean exported poster")
            ax.plot(x, x, linestyle="--", alpha=0.5, label="Identity")
        elif metric_key == "steps":
            y = np.array([r["steps_mean"] for r in per_start_rows])
            ystd = np.array([r["steps_std"] for r in per_start_rows])
            ax.plot(x, y, linewidth=2, label="Mean steps")
        elif metric_key == "export_utility":
            y = np.array([r["export_utility_mean"] for r in per_start_rows])
            ystd = np.array([r["export_utility_std"] for r in per_start_rows])
            ax.plot(x, y, linewidth=2, label="Mean export utility")
        else:
            raise ValueError(f"Unknown metric_key: {metric_key}")

        if np.any(ystd > 0):
            ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")

        ax.set_title(label)
        ax.set_xlabel("Start poster")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    _save_show(fig, save_path=save_path, show=show)

def plot_combined_overlay(results_dict, utility_fn, g_min, g_max, title, save_path=None, show=True):
    labels = list(results_dict.keys())
    n = len(labels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    xs = np.arange(g_min, g_max + 1)
    ys = utility_fn(xs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax in axes[:]:
        ax.grid(alpha=0.3)

    for ax, label in zip(axes, labels):
        result = results_dict[label]
        per_traj_rows = result["per_traj_rows"]
        per_start_rows = result["per_start_rows"]

        ax.plot(xs, ys, color="black", linewidth=2, label="Utility")

        starts = np.array([r["start_poster"] for r in per_traj_rows], dtype=float)
        exports = np.array([r["export_poster"] for r in per_traj_rows], dtype=float)
        ax.scatter(starts, exports, s=8, alpha=0.10, color="tab:blue", label="Trajectories")

        x = np.array([r["start_poster"] for r in per_start_rows], dtype=float)
        y = np.array([r["export_poster_mean"] for r in per_start_rows], dtype=float)
        ax.plot(x, y, color="tab:red", linewidth=2, label="Mean exported poster")

        ax.set_title(label)
        ax.set_xlabel("Start poster")
        ax.set_ylabel("Value")
        ax.legend()

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    _save_show(fig, save_path=save_path, show=show)
    

def plot_q_and_policy(user, g_min=None, g_max=None, title="Q* and Boltzmann policy", save_path=None, show=True):
    if g_min is None:
        g_min = user.g_min
    if g_max is None:
        g_max = user.g_max

    xs = np.arange(g_min, g_max + 1, dtype=int)
    idxs = xs - user.g_min

    q_export = user.Q[EXPORT][idxs]
    q_inc = user.Q[INC][idxs].copy()
    q_dec = user.Q[DEC][idxs].copy()

    q_inc[np.isneginf(q_inc)] = np.nan
    q_dec[np.isneginf(q_dec)] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(xs, q_export, label="Q*(g, EXPORT) = U(g)", linewidth=2)
    ax.plot(xs, q_inc, label="Q*(g, INC) = gamma V(g+1)", linewidth=2)
    ax.plot(xs, q_dec, label="Q*(g, DEC) = gamma V(g-1)", linewidth=2)
    ax.set_ylabel("Q value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    axp = axes[1]
    p_export = np.zeros_like(xs, dtype=float)
    p_inc = np.zeros_like(xs, dtype=float)
    p_dec = np.zeros_like(xs, dtype=float)

    for k, g in enumerate(xs):
        i = g - user.g_min
        actions = [EXPORT]
        if g < user.g_max:
            actions.append(INC)
        if g > user.g_min:
            actions.append(DEC)

        qvals = np.array([user.Q[a][i] for a in actions], dtype=float)
        probs = _softmax_stable(user.alpha * qvals)

        for a, p in zip(actions, probs):
            if a == EXPORT: p_export[k] = p
            if a == INC:    p_inc[k] = p
            if a == DEC:    p_dec[k] = p

    axp.plot(xs, p_export, label="P(EXPORT|g)", linewidth=2)
    axp.plot(xs, p_inc, label="P(INC|g)", linewidth=2)
    axp.plot(xs, p_dec, label="P(DEC|g)", linewidth=2)
    axp.set_xlabel("Poster g")
    axp.set_ylabel("Action probability")
    axp.set_ylim(-0.02, 1.02)
    axp.grid(alpha=0.3)
    axp.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        