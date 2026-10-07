import numpy as np

from .env import DEFAULT_K, Problem2D
from utility import smooth_utility as smooth_utility_1d

PREFERRED_CONTRAST = 5
MAIN_EFFECT_WEIGHT = 0.4
SMOOTH_CONTRAST_WIDTH = 3.0
SMOOTH_GLOBAL_PEAK_WEIGHT = 2.4
SMOOTH_LOCAL_PEAK_WEIGHT = 1.85
SMOOTH_CONTRAST_BAND_WEIGHT = 0.45
SMOOTH_GLOBAL_HEADLINE_CENTER = 8.0
SMOOTH_GLOBAL_BACKGROUND_CENTER = 3.0
SMOOTH_LOCAL_HEADLINE_CENTER = 4.0
SMOOTH_LOCAL_BACKGROUND_CENTER = 9.0
SMOOTH_GLOBAL_HEADLINE_WIDTH = 1.9
SMOOTH_GLOBAL_BACKGROUND_WIDTH = 2.1
SMOOTH_LOCAL_PEAK_WIDTH = 1.55
SMOOTH_LOCAL_CONTRAST_WIDTH = 1.6
SMOOTH_VALLEY_WEIGHT = 0.35
SMOOTH_VALLEY_CENTER = 5.0
SMOOTH_VALLEY_WIDTH = 1.35
SMOOTH_BUMP_WEIGHT = 0.9
SMOOTH_BUMP_HEADLINE_CENTER = 4.0
SMOOTH_BUMP_BACKGROUND_CENTER = 3.0
SMOOTH_BUMP_WIDTH = 0.65

def _validate_k(k):
    if isinstance(k, bool):
        raise ValueError(f"k must be an integer >= 2, got {k!r}.")
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        raise ValueError(f"k must be an integer >= 2, got {k!r}.") from None
    if isinstance(k, float) and not k.is_integer():
        raise ValueError(f"k must be an integer >= 2, got {k!r}.")
    if k_int < 2:
        raise ValueError(f"k must be >= 2 for a contrast utility, got {k!r}.")
    return k_int


def _validate_finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _validate_positive(value, name):
    value = _validate_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _gaussian(x, center, width):
    width = _validate_positive(width, "width")
    return np.exp(-0.5 * ((np.asarray(x, dtype=float) - float(center)) / width) ** 2)


def coord_to_1d_gray(coord, k=DEFAULT_K):
    """
    Map one-based 2D grid coordinates in {1, ..., k} onto the 1D grayscale
    domain {1, ..., 100}.
    """
    k = _validate_k(k)
    arr = np.asarray(coord, dtype=float)

    if np.any(~np.isfinite(arr)):
        raise ValueError("coord must contain only finite values.")
    if np.any((arr < 1.0) | (arr > float(k))):
        raise ValueError(f"coord values must be in [1,{k}].")

    gray = 1.0 + (arr - 1.0) * (99.0 / float(k - 1))
    if np.ndim(coord) == 0:
        return float(gray)
    return gray


def _default_smooth_peak_centers(k, preferred_contrast):
    if k == DEFAULT_K and np.isclose(preferred_contrast, PREFERRED_CONTRAST):
        return {
            "global_headline_center": SMOOTH_GLOBAL_HEADLINE_CENTER,
            "global_background_center": SMOOTH_GLOBAL_BACKGROUND_CENTER,
            "local_headline_center": SMOOTH_LOCAL_HEADLINE_CENTER,
            "local_background_center": SMOOTH_LOCAL_BACKGROUND_CENTER,
        }

    lower = 0.5 * (float(k) - float(preferred_contrast) + 1.0)
    upper = lower + float(preferred_contrast)
    return {
        "global_headline_center": upper,
        "global_background_center": lower,
        "local_headline_center": lower,
        "local_background_center": upper,
    }


def smooth_contrast_interaction_default_params(k=DEFAULT_K):
    """
    Return the default tuning parameters for smooth_contrast_interaction_grid.
    """
    k = _validate_k(k)
    centers = _default_smooth_peak_centers(k, PREFERRED_CONTRAST)
    return {
        "preferred_contrast": float(PREFERRED_CONTRAST),
        "contrast_width": float(SMOOTH_CONTRAST_WIDTH),
        "main_effect_weight": float(MAIN_EFFECT_WEIGHT),
        "background_scale": 0.5,
        "global_peak_weight": float(SMOOTH_GLOBAL_PEAK_WEIGHT),
        "local_peak_weight": float(SMOOTH_LOCAL_PEAK_WEIGHT),
        "contrast_band_weight": float(SMOOTH_CONTRAST_BAND_WEIGHT),
        "global_headline_center": float(centers["global_headline_center"]),
        "global_background_center": float(centers["global_background_center"]),
        "local_headline_center": float(centers["local_headline_center"]),
        "local_background_center": float(centers["local_background_center"]),
        "global_headline_width": float(SMOOTH_GLOBAL_HEADLINE_WIDTH),
        "global_background_width": float(SMOOTH_GLOBAL_BACKGROUND_WIDTH),
        "local_peak_width": float(SMOOTH_LOCAL_PEAK_WIDTH),
        "local_contrast_width": float(SMOOTH_LOCAL_CONTRAST_WIDTH),
        "valley_weight": float(SMOOTH_VALLEY_WEIGHT),
        "valley_center": float(SMOOTH_VALLEY_CENTER),
        "valley_width": float(SMOOTH_VALLEY_WIDTH),
        "bump_weight": float(SMOOTH_BUMP_WEIGHT),
        "bump_headline_center": float(SMOOTH_BUMP_HEADLINE_CENTER),
        "bump_background_center": float(SMOOTH_BUMP_BACKGROUND_CENTER),
        "bump_width": float(SMOOTH_BUMP_WIDTH),
    }


def smooth_contrast_interaction_grid(
    k=DEFAULT_K,
    preferred_contrast=PREFERRED_CONTRAST,
    contrast_width=None,
    main_effect_weight=MAIN_EFFECT_WEIGHT,
    background_scale=0.5,
    global_peak_weight=SMOOTH_GLOBAL_PEAK_WEIGHT,
    local_peak_weight=SMOOTH_LOCAL_PEAK_WEIGHT,
    contrast_band_weight=SMOOTH_CONTRAST_BAND_WEIGHT,
    global_headline_center=None,
    global_background_center=None,
    local_headline_center=None,
    local_background_center=None,
    global_headline_width=SMOOTH_GLOBAL_HEADLINE_WIDTH,
    global_background_width=SMOOTH_GLOBAL_BACKGROUND_WIDTH,
    local_peak_width=SMOOTH_LOCAL_PEAK_WIDTH,
    local_contrast_width=SMOOTH_LOCAL_CONTRAST_WIDTH,
    valley_weight=SMOOTH_VALLEY_WEIGHT,
    valley_center=SMOOTH_VALLEY_CENTER,
    valley_width=SMOOTH_VALLEY_WIDTH,
    bump_weight=SMOOTH_BUMP_WEIGHT,
    bump_headline_center=SMOOTH_BUMP_HEADLINE_CENTER,
    bump_background_center=SMOOTH_BUMP_BACKGROUND_CENTER,
    bump_width=SMOOTH_BUMP_WIDTH,
):
    """
    Build a normalized K x K tuned smooth contrast-interaction utility surface.

    Rows are headline grayscale h and columns are background grayscale b.
    """
    k = _validate_k(k)

    if preferred_contrast is None:
        preferred_contrast = float(k - 1)
    preferred_contrast = _validate_finite(preferred_contrast, "preferred_contrast")
    if preferred_contrast < 0.0 or preferred_contrast > float(k - 1):
        raise ValueError(f"preferred_contrast must be in [0,{k - 1}].")

    if contrast_width is None:
        contrast_width = SMOOTH_CONTRAST_WIDTH
    contrast_width = _validate_finite(contrast_width, "contrast_width")
    if contrast_width <= 0.0:
        raise ValueError(f"contrast_width must be positive, got {contrast_width!r}.")

    main_effect_weight = _validate_finite(main_effect_weight, "main_effect_weight")
    background_scale = _validate_finite(background_scale, "background_scale")
    global_peak_weight = _validate_finite(global_peak_weight, "global_peak_weight")
    local_peak_weight = _validate_finite(local_peak_weight, "local_peak_weight")
    contrast_band_weight = _validate_finite(
        contrast_band_weight,
        "contrast_band_weight",
    )
    local_contrast_width = _validate_positive(
        local_contrast_width,
        "local_contrast_width",
    )
    global_headline_width = _validate_positive(
        global_headline_width,
        "global_headline_width",
    )
    global_background_width = _validate_positive(
        global_background_width,
        "global_background_width",
    )
    local_peak_width = _validate_positive(local_peak_width, "local_peak_width")
    valley_weight = _validate_finite(valley_weight, "valley_weight")
    valley_center = _validate_finite(valley_center, "valley_center")
    valley_width = _validate_positive(valley_width, "valley_width")
    bump_weight = _validate_finite(bump_weight, "bump_weight")
    bump_headline_center = _validate_finite(
        bump_headline_center,
        "bump_headline_center",
    )
    bump_background_center = _validate_finite(
        bump_background_center,
        "bump_background_center",
    )
    bump_width = _validate_positive(bump_width, "bump_width")

    default_centers = _default_smooth_peak_centers(k, preferred_contrast)
    if global_headline_center is None:
        global_headline_center = default_centers["global_headline_center"]
    if global_background_center is None:
        global_background_center = default_centers["global_background_center"]
    if local_headline_center is None:
        local_headline_center = default_centers["local_headline_center"]
    if local_background_center is None:
        local_background_center = default_centers["local_background_center"]

    global_headline_center = _validate_finite(
        global_headline_center,
        "global_headline_center",
    )
    global_background_center = _validate_finite(
        global_background_center,
        "global_background_center",
    )
    local_headline_center = _validate_finite(
        local_headline_center,
        "local_headline_center",
    )
    local_background_center = _validate_finite(
        local_background_center,
        "local_background_center",
    )

    coords = np.arange(1, k + 1, dtype=float)
    H, B = np.meshgrid(coords, coords, indexing="ij")

    contrast = np.abs(H - B)
    contrast_gate = _gaussian(contrast, preferred_contrast, contrast_width)
    local_contrast_gate = _gaussian(contrast, preferred_contrast, local_contrast_width)

    headline_gray = coord_to_1d_gray(H, k=k)
    background_gray = coord_to_1d_gray(B, k=k)

    global_peak = (
        global_peak_weight
        * _gaussian(H, global_headline_center, global_headline_width)
        * _gaussian(B, global_background_center, global_background_width)
        * contrast_gate
    )
    local_peak = (
        local_peak_weight
        * _gaussian(H, local_headline_center, local_peak_width)
        * _gaussian(B, local_background_center, local_peak_width)
        * local_contrast_gate
    )
    contrast_band = contrast_band_weight * contrast_gate
    main_effect = main_effect_weight * (
        smooth_utility_1d(headline_gray)
        + background_scale * smooth_utility_1d(background_gray)
    )
    valley = (
        -valley_weight
        * _gaussian(H, valley_center, valley_width)
        * _gaussian(B, valley_center, valley_width)
    )
    bump = (
        bump_weight
        * _gaussian(H, bump_headline_center, bump_width)
        * _gaussian(B, bump_background_center, bump_width)
    )

    raw = global_peak + local_peak + contrast_band + main_effect + valley + bump
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    if np.isclose(raw_min, raw_max):
        raise ValueError("Cannot normalize a constant 2D utility surface.")

    return (raw - raw_min) / (raw_max - raw_min)


def smooth_contrast_interaction_utility(
    posters,
    k=DEFAULT_K,
    preferred_contrast=PREFERRED_CONTRAST,
    contrast_width=None,
    main_effect_weight=MAIN_EFFECT_WEIGHT,
    background_scale=0.5,
    global_peak_weight=SMOOTH_GLOBAL_PEAK_WEIGHT,
    local_peak_weight=SMOOTH_LOCAL_PEAK_WEIGHT,
    contrast_band_weight=SMOOTH_CONTRAST_BAND_WEIGHT,
    global_headline_center=None,
    global_background_center=None,
    local_headline_center=None,
    local_background_center=None,
    global_headline_width=SMOOTH_GLOBAL_HEADLINE_WIDTH,
    global_background_width=SMOOTH_GLOBAL_BACKGROUND_WIDTH,
    local_peak_width=SMOOTH_LOCAL_PEAK_WIDTH,
    local_contrast_width=SMOOTH_LOCAL_CONTRAST_WIDTH,
    valley_weight=SMOOTH_VALLEY_WEIGHT,
    valley_center=SMOOTH_VALLEY_CENTER,
    valley_width=SMOOTH_VALLEY_WIDTH,
    bump_weight=SMOOTH_BUMP_WEIGHT,
    bump_headline_center=SMOOTH_BUMP_HEADLINE_CENTER,
    bump_background_center=SMOOTH_BUMP_BACKGROUND_CENTER,
    bump_width=SMOOTH_BUMP_WIDTH,
):
    """
    Poster-id callable for the smooth contrast-interaction utility.
    """
    problem = Problem2D(k=k)
    grid = smooth_contrast_interaction_grid(
        k=k,
        preferred_contrast=preferred_contrast,
        contrast_width=contrast_width,
        main_effect_weight=main_effect_weight,
        background_scale=background_scale,
        global_peak_weight=global_peak_weight,
        local_peak_weight=local_peak_weight,
        contrast_band_weight=contrast_band_weight,
        global_headline_center=global_headline_center,
        global_background_center=global_background_center,
        local_headline_center=local_headline_center,
        local_background_center=local_background_center,
        global_headline_width=global_headline_width,
        global_background_width=global_background_width,
        local_peak_width=local_peak_width,
        local_contrast_width=local_contrast_width,
        valley_weight=valley_weight,
        valley_center=valley_center,
        valley_width=valley_width,
        bump_weight=bump_weight,
        bump_headline_center=bump_headline_center,
        bump_background_center=bump_background_center,
        bump_width=bump_width,
    )

    poster_arr = np.asarray(posters)
    scalar_input = poster_arr.ndim == 0
    flat_posters = poster_arr.reshape(-1)

    values = []
    for poster in flat_posters:
        h, b = problem.from_poster(poster)
        values.append(float(grid[h - 1, b - 1]))

    out = np.asarray(values, dtype=float).reshape(poster_arr.shape)
    if scalar_input:
        return float(out)
    return out
