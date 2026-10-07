import numpy as np
# import matplotlib.pyplot as plt

def raw_utility(g):
    g = np.asarray(g, dtype=float)
    term_near = 2.6 * np.exp(-((g - 20.0) ** 2) / (2.0 * (4.0 ** 2)))
    term_far = 4.0 * np.exp(-((g - 85.0) ** 2) / (2.0 * (7.0 ** 2)))
    valley = -2.2 * np.exp(-((g - 45.0) ** 2) / (2.0 * (9.0 ** 2)))
    ripple = 0.25 * np.sin(2.0 * np.pi * g / 18.0)
    return term_near + term_far + valley + ripple


def normalize_utility(utility_fn, g_min=1, g_max=100):
    grid = np.arange(g_min, g_max + 1, dtype=float)
    values = np.asarray(utility_fn(grid), dtype=float)

    umin = float(np.min(values))
    umax = float(np.max(values))
    if np.isclose(umax, umin):
        raise ValueError("Cannot normalize a constant utility function.")

    def normalized(g):
        u = np.asarray(utility_fn(g), dtype=float)
        return (u - umin) / (umax - umin)

    return normalized


def jagged_raw_utility(
    base_raw_fn,
    g_min=1,
    g_max=100,
    seed=7,
    rough_strength=0.05,
    alternating_strength=0.015,
    spike_strength=0.040,
    peak_anchor_strength=0.050,
):
    grid = np.arange(g_min, g_max + 1, dtype=float)
    base_values = np.asarray(base_raw_fn(grid), dtype=float)
    scale = np.ptp(base_values)

    rng = np.random.default_rng(seed)

    noise = rng.normal(0.0, 1.0, size=len(grid))
    kernel = np.array([0.15, 0.70, 0.15])
    rough = np.convolve(noise, kernel, mode="same")
    rough = rough - np.mean(rough)
    rough = rough / np.std(rough)
    rough = rough / np.max(np.abs(rough))

    idx = np.arange(len(grid))
    alternating = (-1.0) ** idx


    spike_locs = np.array([8, 14, 27, 33, 52, 59, 73, 79, 91, 97], dtype=float)
    spike_amps = np.array([0.7, -0.5, 0.6, -0.8, 0.5, -0.7, 0.8, -0.4, -0.7, 0.5])

    spikes = np.zeros_like(grid)
    for loc, amp in zip(spike_locs, spike_amps):
        spikes += amp * np.exp(-((grid - loc) ** 2) / (2.0 * (0.9 ** 2)))

    spikes = spikes - np.mean(spikes)
    spikes = spikes / np.max(np.abs(spikes))

    peak_anchor = np.exp(-((grid - 85.0) ** 2) / (2.0 * (1.5 ** 2)))

    jagged_values = (
        base_values
        + scale * rough_strength * rough
        + scale * alternating_strength * alternating
        + scale * spike_strength * spikes
        + scale * peak_anchor_strength * peak_anchor
    )

    def jagged_raw(g):
        g = np.asarray(g, dtype=float)
        return np.interp(g, grid, jagged_values)

    return jagged_raw


utility = normalize_utility(raw_utility, g_min=1, g_max=100)
smooth_utility = utility

raw_jagged_utility = jagged_raw_utility(raw_utility, seed=1)
jagged_utility = normalize_utility(raw_jagged_utility, g_min=1, g_max=100)


VALID_UTILITY_KINDS = ("smooth", "jagged")


def canonical_utility_kind(kind: str) -> str:
    k = str(kind).strip().lower()

    aliases = {
        "smooth": "smooth",
        "base": "smooth",
        "original": "smooth",

        "jagged": "jagged",
        "rough": "jagged",
    }

    if k not in aliases:
        raise ValueError(
            f"Unknown utility kind {kind!r}. "
            f"Expected one of {VALID_UTILITY_KINDS}."
        )

    return aliases[k]


def get_utility(kind: str):
    k = canonical_utility_kind(kind)

    if k == "smooth":
        return smooth_utility
    if k == "jagged":
        return jagged_utility

    raise AssertionError(f"Unhandled canonical utility kind: {k}")


def get_utility_label(kind: str) -> str:
    k = canonical_utility_kind(kind)
    if k == "smooth":
        return "Smooth utility"
    if k == "jagged":
        return "Jagged utility"
    raise AssertionError(f"Unhandled canonical utility kind: {k}")

# G = np.arange(1, 101)
# F = jagged_utility(G)
# g_star = G[np.argmax(F)]

# g_fine = np.linspace(1, 100, 2000)
# f_fine = jagged_utility(g_fine)

# plt.figure(figsize=(9, 4.5))
# plt.plot(g_fine, f_fine)
# plt.scatter(G, F, s=18, alpha=0.7)

# plt.axvline(g_star, linestyle="--", linewidth=1, label=f"Global max at {g_star}")
# plt.scatter([g_star], [F.max()], zorder=5)

# plt.title("")
# plt.xlabel("Poster")
# plt.ylabel("Utility")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# print("Grid G from", G[0], "to", G[-1], "with", len(G), "states")
# print("g* =", g_star)
# print("max f =", float(F.max()))