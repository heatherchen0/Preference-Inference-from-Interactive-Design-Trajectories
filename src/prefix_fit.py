import json
import hashlib
from pathlib import Path
import numpy as np
from .model import (
    build_model2_discounted,
    build_model4_endpoint_discounted_boltzmann,
    build_model4_endpoint_discounted_boltzmann_infer_alpha,
    build_model4_endpoint_discounted_boltzmann_2d,
    build_birl_baseline_discounted_boltzmann,
    build_birl_baseline_discounted_boltzmann_2d,
    build_pbo_pairwise_preference_model,
)
from .sampling import sample_model
from .preprocess import (
    prepare_inverse_data,
    prepare_pbo_start_end_data,
    prepare_pbo_transition_data,
)
import pandas as pd
from .elpd import compute_elpd_for_df, first_n_trajectories 
from .metrics import (
    summarize_posterior_var,
    summarize_soft_copeland_from_idata,
    compute_posterior_pairwise_accuracy_from_idata,
)
from .state_space import action_order, normalize_state_space, normalized_gp_inputs
from .posterior_archive import (
    save_alpha_draw_archive,
    save_u_draw_archive,
    summarize_idata_diagnostics,
    write_rows_csv,
)


_PBO_START_END_MODELS = ("pbo_baseline", "pbo_baseline_start_end_preference")
_PBO_TRANSITION_MODELS = ("pbo_baseline_transition_preference",)
_PBO_MODELS = _PBO_START_END_MODELS + _PBO_TRANSITION_MODELS


def _is_pbo_model_name(model_name: str) -> bool:
    return str(model_name).lower() in _PBO_MODELS


def _infers_alpha_model(model_name: str) -> bool:
    return str(model_name).lower() == "model4_endpoint_discounted_boltzmann_infer_alpha"


def _posterior_alpha_summary(
    idata,
    *,
    hdi_prob: float,
    alpha_prior_median: float,
    alpha_prior_log_sd: float,
) -> dict | None:
    if not hasattr(idata, "posterior") or "alpha" not in idata.posterior:
        return None
    out = summarize_posterior_var(
        idata,
        var_name="alpha",
        hdi_prob=float(hdi_prob),
        normalize_draws=False,
    )
    def _scalar(x):
        return float(np.asarray(x, dtype=float).reshape(-1)[0])
    draws = np.asarray(idata.posterior["alpha"].values, dtype=float).reshape(-1)

    return {
        "mean": _scalar(out["mean"]),
        "median": float(np.nanmedian(draws)),
        "sd": _scalar(out["sd"]),
        "q05": float(np.nanquantile(draws, 0.05)),
        "q95": float(np.nanquantile(draws, 0.95)),
        "low": _scalar(out["low"]),
        "high": _scalar(out["high"]),
        "hdi_prob": float(out["hdi_prob"]),
        "var_name": "alpha",
        "prior": "LogNormal",
        "prior_median": float(alpha_prior_median),
        "prior_log_sd": float(alpha_prior_log_sd),
    }


def _pbo_duel_label(model_name: str) -> str:
    name = str(model_name).lower()
    if name in _PBO_TRANSITION_MODELS:
        return "transition"
    if name in _PBO_START_END_MODELS:
        return "start-end"
    raise ValueError(f"Unknown PBO model_name={model_name!r}")


def _prepare_pbo_data_for_model(
    model_name: str,
    data,
    *,
    poster_min: int,
    poster_max: int,
    state_space: str = "1d",
    grid_k: int | None = None,
):
    if _pbo_duel_label(model_name) == "transition":
        return prepare_pbo_transition_data(
            data,
            poster_min=poster_min,
            poster_max=poster_max,
            drop_ties=True,
            state_space=state_space,
            grid_k=grid_k,
        )
    return prepare_pbo_start_end_data(
        data,
        poster_min=poster_min,
        poster_max=poster_max,
        drop_ties=True,
        state_space=state_space,
        grid_k=grid_k,
    )

def build_prefix_k_traj_list(
    n_traj: int,
    *,
    schedule: str = "geom2",
    traj_chunk_size: int = 300,
    geom_start: int = 8,
    geom_ratio: int = 2,
    k_traj_list: list[int] | None = None,
) -> list[int]:
    """
    Build a list of cumulative-trajectory counts to run prefix fits at.

    schedule:
    - "geom2": geom_start, geom_start*geom_ratio, ..., n_traj (always includes n_traj)
    - "fixed_step": traj_chunk_size, 2*traj_chunk_size, ..., n_traj
    - "explicit": use k_traj_list only (still forces inclusion of n_traj)

    for non-"explicit" schedules, k_traj_list is UNIONED in
    """
    n_traj = int(n_traj)
    if n_traj <= 0:
        raise ValueError(f"n_traj must be >= 1, got {n_traj}")

    schedule = str(schedule).lower().strip()
    ks: list[int] = []

    if schedule in ("explicit", "list"):
        ks.extend([] if k_traj_list is None else [int(k) for k in k_traj_list])

    elif schedule in ("geom", "geom2", "geometric"):
        start = int(geom_start)
        ratio = int(geom_ratio)
        if start <= 0:
            raise ValueError(f"geom_start must be >= 1, got {start}")
        if ratio <= 1:
            raise ValueError(f"geom_ratio must be >= 2, got {ratio}")

        k = start
        while k < n_traj:
            ks.append(int(k))
            k *= ratio

        if k_traj_list:
            ks.extend([int(k) for k in k_traj_list])

    elif schedule in ("fixed", "fixed_step", "linear"):
        step = int(traj_chunk_size)
        if step <= 0:
            raise ValueError(
                f"traj_chunk_size must be >= 1 for fixed_step schedule, got {step}"
            )
        ks.extend(list(range(step, n_traj + 1, step)))
        if k_traj_list:
            ks.extend([int(k) for k in k_traj_list])

    else:
        raise ValueError(
            f"Unknown prefix schedule={schedule!r} "
            "(use 'geom2', 'fixed_step', or 'explicit')."
        )

    ks = [int(k) for k in ks if k is not None]
    ks = [k for k in ks if 1 <= k <= n_traj]
    ks = sorted(set(ks))
    if len(ks) == 0 or ks[-1] != n_traj:
        ks.append(n_traj)
    return ks


def posterior_U_summary(idata, hdi_prob: float = 0.94, normalize_draws: bool = False):
    s = summarize_posterior_var(
        idata,
        var_name="U",
        hdi_prob=float(hdi_prob),
        normalize_draws=bool(normalize_draws),
    )
    return (
        np.asarray(s["mean"]),
        np.asarray(s["sd"]),
        np.asarray(s["low"]),
        np.asarray(s["high"]),
    )


def _cache_path(run_dir: Path) -> Path:
    return Path(run_dir) / "prefix_summaries.json"


def _jsonable_id(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _jsonable_ids(xs):
    if xs is None:
        return None
    return [_jsonable_id(x) for x in list(xs)]


def _numeric_vector_signature(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = np.ascontiguousarray(arr)
    return {
        "n": int(arr.size),
        "sha256": hashlib.sha256(arr.view(np.uint8)).hexdigest(),
    }


def _traj_sort_key(x):
    try:
        return (0, int(x))
    except Exception:
        return (1, str(x))


def _default_sorted_traj_ids(raw_ids):
    return sorted(list(raw_ids), key=_traj_sort_key)


def _resolve_trajectory_order(raw_ids, trajectory_order=None):
    """
    Resolve the trajectory order used for prefix fitting.

    If trajectory_order is None, fall back to sorted traj_id order.
    If provided, validate that it is a permutation of the trajectory ids in df.
    """
    raw_ids = list(raw_ids)

    if len(raw_ids) == 0:
        raise ValueError("No trajectories found: traj_id is empty.")

    if trajectory_order is None:
        return _default_sorted_traj_ids(raw_ids)

    order = list(trajectory_order)

    if len(order) != len(raw_ids):
        raise ValueError(
            "trajectory_order length mismatch: "
            f"got {len(order)}, expected {len(raw_ids)}."
        )

    raw_set = set(raw_ids)
    order_set = set(order)

    if raw_set != order_set:
        missing = sorted(list(raw_set - order_set), key=_traj_sort_key)
        extra = sorted(list(order_set - raw_set), key=_traj_sort_key)
        raise ValueError(
            "trajectory_order must be a permutation of df['traj_id']. "
            f"Missing ids: {missing[:20]} Extra ids: {extra[:20]}"
        )

    if len(order_set) != len(order):
        raise ValueError("trajectory_order contains duplicate trajectory ids.")

    return order


def _cache_key(
    traj_ids,
    poster_min,
    poster_max,
    gamma_fixed,
    beta_fixed,
    alpha_fixed,
    alpha_prior_median,
    alpha_prior_log_sd,
    model_name,
    likelihood,
    prior,
    gp_eta_sd,
    gp_ls_sd,
    draws,
    tune,
    chains,
    cores,
    target_accept,
    random_seed,
    k_traj_list,
    hdi_prob,
    eval_seen_traj_ids=None,
    eval_seen_test_traj_chunk_size: int | None = None,
    reuse_main_idata_for_full_prefix: bool = False,
    trajectory_order=None,
    trajectory_permutation_seed: int | None = None,
    U_true_for_metrics=None,
    posters_for_archive=None,
    posterior_archive_dir: str | Path | None = None,
    posterior_archive_metadata: dict | None = None,
    archive_dtype: str = "float32",
    save_u_draw_archives: bool = False,
    save_alpha_draw_archives: bool = False,
    save_diagnostics: bool = False,
    main_u_archive_path: str | None = None,
    main_alpha_archive_path: str | None = None,
    save_full_idata_archive: bool = False,
    main_idata_archive_path: str | None = None,
    birl_export_mode: str = "stay",
    birl_reward_scale: str = "discounted",
    birl_value_iters: int = 1000,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
):
    payload = {
        "state_space": str(normalize_state_space(state_space)),
        "grid_k": None if grid_k is None else int(grid_k),
        "action_order": list(action_order(state_space)),
        "preferred_contrast": (
            None if preferred_contrast is None else float(preferred_contrast)
        ),
        "model_name": str(model_name),
        "likelihood": str(likelihood),
        "prior": str(prior),
        "gp_eta_sd": float(gp_eta_sd),
        "gp_ls_sd": float(gp_ls_sd),

        # traj_ids is the actual ordered prefix trajectory list.
        "traj_ids": _jsonable_ids(traj_ids),

        # Explicitly store the external trajectory order and seed too.
        "trajectory_order": _jsonable_ids(trajectory_order),
        "trajectory_permutation_seed": (
            int(trajectory_permutation_seed)
            if trajectory_permutation_seed is not None
            else None
        ),
        "U_true_for_metrics": _numeric_vector_signature(U_true_for_metrics),
        "save_u_draw_archives": bool(save_u_draw_archives),
        "save_alpha_draw_archives": bool(save_alpha_draw_archives),
        "save_diagnostics": bool(save_diagnostics),
        "save_full_idata_archive": bool(save_full_idata_archive),
        "archive_dtype": str(archive_dtype),

        "poster_min": int(poster_min),
        "poster_max": int(poster_max),
        "gamma_fixed": float(gamma_fixed),
        "beta_fixed": float(beta_fixed),
        "alpha_fixed": float(alpha_fixed),
        "alpha_prior_median": float(alpha_prior_median),
        "alpha_prior_log_sd": float(alpha_prior_log_sd),
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "cores": int(cores),
        "target_accept": float(target_accept),
        "random_seed": int(random_seed),
        "k_traj_list": [int(k) for k in k_traj_list],
        "hdi_prob": float(hdi_prob),
        "eval_seen_traj_ids": _jsonable_ids(eval_seen_traj_ids),
        "eval_seen_test_traj_chunk_size": (
            int(eval_seen_test_traj_chunk_size)
            if eval_seen_test_traj_chunk_size is not None
            else None
        ),
        "reuse_main_idata_for_full_prefix": bool(reuse_main_idata_for_full_prefix),
        "birl_export_mode": str(birl_export_mode),
        "birl_reward_scale": str(birl_reward_scale),
        "birl_value_iters": int(birl_value_iters),

        "schema": 17,
    }
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def save_prefix_summaries(prefix_summaries, path: Path, cache_key: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "cache_key": cache_key,
        "prefix_summaries": [
            {
                **{k: v for k, v in s.items() if k not in ("mean", "sd", "low", "high")},
                "mean": np.asarray(s["mean"], dtype=float).tolist(),
                "sd": np.asarray(s["sd"], dtype=float).tolist(),
                "low": np.asarray(s["low"], dtype=float).tolist(),
                "high": np.asarray(s["high"], dtype=float).tolist(),
                "mean_norm_draws": np.asarray(s.get("mean_norm_draws", []), dtype=float).tolist(),
                "sd_norm_draws": np.asarray(s.get("sd_norm_draws", []), dtype=float).tolist(),
                "low_norm_draws": np.asarray(s.get("low_norm_draws", []), dtype=float).tolist(),
                "high_norm_draws": np.asarray(s.get("high_norm_draws", []), dtype=float).tolist(),
                "soft_copeland_mean": np.asarray(s.get("soft_copeland_mean", []), dtype=float).tolist(),
                "soft_copeland_sd": np.asarray(s.get("soft_copeland_sd", []), dtype=float).tolist(),
                "soft_copeland_low": np.asarray(s.get("soft_copeland_low", []), dtype=float).tolist(),
                "soft_copeland_high": np.asarray(s.get("soft_copeland_high", []), dtype=float).tolist(),
            }
            for s in prefix_summaries
        ],
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_prefix_summaries(path: Path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["cache_key"], [
        {
            **{k: v for k, v in s.items() if k not in ("mean", "sd", "low", "high")},
            "mean": np.asarray(s["mean"], dtype=float),
            "sd": np.asarray(s["sd"], dtype=float),
            "low": np.asarray(s["low"], dtype=float),
            "high": np.asarray(s["high"], dtype=float),
            **(
                {
                    "mean_norm_draws": np.asarray(s["mean_norm_draws"], dtype=float),
                    "sd_norm_draws": np.asarray(s["sd_norm_draws"], dtype=float),
                    "low_norm_draws": np.asarray(s["low_norm_draws"], dtype=float),
                    "high_norm_draws": np.asarray(s["high_norm_draws"], dtype=float),
                }
                if "mean_norm_draws" in s
                else {}
            ),
            **(
                {
                    "soft_copeland_mean": np.asarray(s["soft_copeland_mean"], dtype=float),
                    "soft_copeland_sd": np.asarray(s["soft_copeland_sd"], dtype=float),
                    "soft_copeland_low": np.asarray(s["soft_copeland_low"], dtype=float),
                    "soft_copeland_high": np.asarray(s["soft_copeland_high"], dtype=float),
                }
                if "soft_copeland_mean" in s
                and len(s.get("soft_copeland_mean", [])) > 0
                else {}
            ),
        }
        for s in raw["prefix_summaries"]
    ]
    
    
def _observed_n_from_idata(idata, obs_var_name: str = "obs") -> int | None:
    try:
        if hasattr(idata, "observed_data") and obs_var_name in idata.observed_data:
            return int(idata.observed_data[obs_var_name].size)
    except Exception:
        return None
    return None


def fit_prefixes_by_trajectory(
    df,
    poster_min,
    poster_max,
    gamma_fixed,
    beta_fixed,
    alpha_fixed,
    model_name,
    alpha_prior_median=5.0,
    alpha_prior_log_sd=0.75,
    likelihood="argmax",
    prior="rw",
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    draws=400,
    tune=400,
    chains=2,
    cores=1,
    target_accept=0.95,
    random_seed=42,
    traj_chunk_size: int = 300,
    hdi_prob: float = 0.94,
    store_traj_ids: bool = True,
    eval_seen_df: pd.DataFrame | None = None,
    eval_seen_test_traj_chunk_size: int = 30,
    k_traj_list: list[int] | None = None,
    main_idata=None,
    reuse_main_idata_for_full_prefix: bool = False,
    trajectory_order=None,
    trajectory_permutation_seed: int | None = None,
    U_true_for_metrics=None,
    posters_for_archive=None,
    posterior_archive_dir: str | Path | None = None,
    posterior_archive_metadata: dict | None = None,
    archive_dtype: str = "float32",
    save_u_draw_archives: bool = False,
    save_alpha_draw_archives: bool = False,
    save_diagnostics: bool = False,
    main_u_archive_path: str | None = None,
    main_alpha_archive_path: str | None = None,
    save_full_idata_archive: bool = False,
    main_idata_archive_path: str | None = None,
    birl_export_mode: str = "stay",
    birl_reward_scale: str = "discounted",
    birl_value_iters: int = 1000,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
):
    state_space = normalize_state_space(state_space)
    required = {"poster", "action", "traj_id", "t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fit_prefixes_by_trajectory: missing required columns: {missing}")

    if k_traj_list is None and int(traj_chunk_size) <= 0:
        raise ValueError(f"traj_chunk_size must be >= 1, got {traj_chunk_size}")

    df = df.sort_values(["traj_id", "t"]).reset_index(drop=True)
    
    if eval_seen_df is not None:
        eval_seen_df = eval_seen_df.sort_values(["traj_id", "t"]).reset_index(drop=True)

    raw_ids = df["traj_id"].unique().tolist()
    traj_ids = _resolve_trajectory_order(
        raw_ids,
        trajectory_order=trajectory_order,
    )

    if len(traj_ids) == 0:
        raise ValueError("fit_prefixes_by_trajectory: no trajectories found (traj_id is empty).")

    num_states = int(poster_max - poster_min + 1)
    gp_inputs = (
        normalized_gp_inputs(
            state_space=state_space,
            num_states=num_states,
            grid_k=grid_k,
        )
        if str(prior).lower() in {"gp", "gp_rbf", "rbf", "logistic_gp", "logistic_gp_rbf", "gp_uniform", "gp_uniform_rbf", "gp_copula", "gp_copula_rbf", "copula_gp", "copula_gp_rbf"}
        else None
    )
    if U_true_for_metrics is not None:
        U_true_for_metrics = np.asarray(U_true_for_metrics, dtype=float).reshape(-1)
        if U_true_for_metrics.shape[0] != num_states:
            raise ValueError(
                "U_true_for_metrics length mismatch: "
                f"got {U_true_for_metrics.shape[0]}, expected {num_states}."
            )
    posters_for_archive = (
        np.arange(int(poster_min), int(poster_max) + 1, dtype=int)
        if posters_for_archive is None
        else np.asarray(posters_for_archive, dtype=int).reshape(-1)
    )
    if posters_for_archive.shape[0] != num_states:
        raise ValueError(
            f"posters_for_archive length mismatch: got {posters_for_archive.shape[0]}, expected {num_states}."
        )
    posterior_archive_dir = (
        None if posterior_archive_dir is None else Path(posterior_archive_dir)
    )
    prefix_diag_rows = []
    prefix_alpha_diag_rows = []

    n_traj = len(traj_ids)
    step = int(traj_chunk_size)
    if k_traj_list is None:
        ks = build_prefix_k_traj_list(
            n_traj=n_traj,
            schedule="geom2",
            traj_chunk_size=step,
            geom_start=8,
            geom_ratio=2,
            k_traj_list=None,
        )
    else:
        ks = build_prefix_k_traj_list(
            n_traj=n_traj,
            schedule="explicit",
            traj_chunk_size=step,
            geom_start=8,
            geom_ratio=2,
            k_traj_list=k_traj_list,
        )

    def _traj_start_poster(_df, tid) -> int:
        dft = _df[_df["traj_id"] == tid]
        if len(dft) == 0:
            return -1
        r0 = dft.loc[dft["t"].astype(int).idxmin()]
        return int(r0["poster"])

    prefix_summaries = []
    if bool(reuse_main_idata_for_full_prefix) and main_idata is None:
        raise ValueError(
            "reuse_main_idata_for_full_prefix=True but main_idata=None. "
            "Pass the main full-data idata, or set reuse_main_idata_for_full_prefix=False."
        )

    for prefix_idx, k_traj in enumerate(ks, start=1):
        keep_ids = traj_ids[:k_traj]
        df_k = df[df["traj_id"].isin(keep_ids)].copy()

        state_idx, action_idx, num_states_k = prepare_inverse_data(
            df_k,
            poster_min=poster_min,
            poster_max=poster_max,
            state_space=state_space,
        )
        if num_states_k != num_states:
            raise RuntimeError(
                f"Unexpected num_states change at k_traj={k_traj}: {num_states_k} vs {num_states}"
            )

        prev_k = 0 if prefix_idx == 1 else ks[prefix_idx - 2]
        new_ids = traj_ids[prev_k:k_traj]
        new_start_posters = sorted({ _traj_start_poster(df, tid) for tid in new_ids })

        is_full_prefix = int(k_traj) == int(n_traj)
        reuse_main_idata = bool(reuse_main_idata_for_full_prefix) and is_full_prefix

        if reuse_main_idata:
            if _is_pbo_model_name(model_name):
                _pbo_start, _pbo_end, _pbo_states, pbo_stats_reuse, _pbo_cmp = _prepare_pbo_data_for_model(
                    model_name,
                    df_k,
                    poster_min=poster_min,
                    poster_max=poster_max,
                    state_space=state_space,
                    grid_k=grid_k,
                )
                expected_observed_n = int(pbo_stats_reuse["n_pbo_duels"])
            else:
                expected_observed_n = int(len(df_k))

            observed_n = _observed_n_from_idata(main_idata, obs_var_name="obs")
            if observed_n is not None and observed_n != expected_observed_n:
                raise ValueError(
                    "Requested reuse of main_idata for the full prefix, but the observed-data "
                    f"length does not match. main_idata observed_n={observed_n}, "
                    f"full-prefix expected observed_n={expected_observed_n}."
                )

            idata_k = main_idata

        else:
            name = str(model_name).lower()
            if name == "model2_discounted":
                model_k = build_model2_discounted(
                    state_idx=state_idx,
                    action_idx=action_idx,
                    num_states=num_states,
                    gamma_fixed=float(gamma_fixed),
                    beta_fixed=float(beta_fixed),
                    likelihood=str(likelihood).lower(),
                    eps_fixed=0.01,
                    infer_eps=False,
                    prior=str(prior).lower(),
                    gp_eta_sd=float(gp_eta_sd),
                    gp_ls_sd=float(gp_ls_sd),
                )
            elif name == "model4_endpoint_discounted_boltzmann":
                if state_space == "2d":
                    model_k = build_model4_endpoint_discounted_boltzmann_2d(
                        state_idx=state_idx,
                        action_idx=action_idx,
                        num_states=num_states,
                        grid_k=int(grid_k),
                        gamma_fixed=float(gamma_fixed),
                        alpha_fixed=float(alpha_fixed),
                        prior=str(prior).lower(),
                        gp_eta_sd=float(gp_eta_sd),
                        gp_ls_sd=float(gp_ls_sd),
                        gp_inputs=gp_inputs,
                    )
                else:
                    model_k = build_model4_endpoint_discounted_boltzmann(
                        state_idx=state_idx,
                        action_idx=action_idx,
                        num_states=num_states,
                        gamma_fixed=float(gamma_fixed),
                        alpha_fixed=float(alpha_fixed),
                        prior=str(prior).lower(),
                        gp_eta_sd=float(gp_eta_sd),
                        gp_ls_sd=float(gp_ls_sd),
                        gp_inputs=gp_inputs,
                    )
            elif name == "model4_endpoint_discounted_boltzmann_infer_alpha":
                if state_space != "1d":
                    raise ValueError(
                        "model4_endpoint_discounted_boltzmann_infer_alpha is v1 1D-only; "
                        f"got state_space={state_space!r}."
                    )
                model_k = build_model4_endpoint_discounted_boltzmann_infer_alpha(
                    state_idx=state_idx,
                    action_idx=action_idx,
                    num_states=num_states,
                    gamma_fixed=float(gamma_fixed),
                    alpha_prior_median=float(alpha_prior_median),
                    alpha_prior_log_sd=float(alpha_prior_log_sd),
                    prior=str(prior).lower(),
                    gp_eta_sd=float(gp_eta_sd),
                    gp_ls_sd=float(gp_ls_sd),
                    gp_inputs=gp_inputs,
                )
            elif name in ("birl_baseline", "birl_baseline_discounted_boltzmann"):
                if state_space == "2d":
                    model_k = build_birl_baseline_discounted_boltzmann_2d(
                        state_idx=state_idx,
                        action_idx=action_idx,
                        num_states=num_states,
                        grid_k=int(grid_k),
                        gamma_fixed=float(gamma_fixed),
                        alpha_fixed=float(alpha_fixed),
                        prior=str(prior).lower(),
                        export_mode=str(birl_export_mode),
                        reward_scale=str(birl_reward_scale),
                        value_iters=int(birl_value_iters),
                        gp_eta_sd=float(gp_eta_sd),
                        gp_ls_sd=float(gp_ls_sd),
                        gp_inputs=gp_inputs,
                    )
                else:
                    model_k = build_birl_baseline_discounted_boltzmann(
                        state_idx=state_idx,
                        action_idx=action_idx,
                        num_states=num_states,
                        gamma_fixed=float(gamma_fixed),
                        alpha_fixed=float(alpha_fixed),
                        prior=str(prior).lower(),
                        export_mode=str(birl_export_mode),
                        reward_scale=str(birl_reward_scale),
                        value_iters=int(birl_value_iters),
                        gp_eta_sd=float(gp_eta_sd),
                        gp_ls_sd=float(gp_ls_sd),
                        gp_inputs=gp_inputs,
                    )
            elif name in _PBO_MODELS:
                start_idx, end_idx, num_states_pbo, _pbo_stats, _pbo_cmp = _prepare_pbo_data_for_model(
                    name,
                    df_k,
                    poster_min=poster_min,
                    poster_max=poster_max,
                    state_space=state_space,
                    grid_k=grid_k,
                )
                if num_states_pbo != num_states:
                    raise RuntimeError(
                        f"Unexpected num_states change for PBO at k_traj={k_traj}: "
                        f"{num_states_pbo} vs {num_states}"
                    )
                model_k = build_pbo_pairwise_preference_model(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_states=num_states,
                    prior=str(prior).lower(),
                    gp_eta_sd=float(gp_eta_sd),
                    gp_ls_sd=float(gp_ls_sd),
                    gp_inputs=gp_inputs,
                )
        
            else:
                raise ValueError(f"Unknown model_name={model_name!r}")

            idata_k = sample_model(
                model_k,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
            )

        raw_U = summarize_posterior_var(
            idata_k,
            var_name="U",
            hdi_prob=float(hdi_prob),
            normalize_draws=False,
        )
        norm_U = summarize_posterior_var(
            idata_k,
            var_name="U",
            hdi_prob=float(hdi_prob),
            normalize_draws=True,
        )
        alpha_summary = _posterior_alpha_summary(
            idata_k,
            hdi_prob=float(hdi_prob),
            alpha_prior_median=float(alpha_prior_median),
            alpha_prior_log_sd=float(alpha_prior_log_sd),
        )

        pbo_stats = {
            "pbo_comparison_type": "",
            "n_pbo_duels": 0,
            "n_ties_dropped": 0,
            "n_start_end_ties": 0,
            "n_transition_ties": 0,
            "n_transition_candidates": 0,
            "tie_fraction": float("nan"),
        }
        if _is_pbo_model_name(model_name):
            _pbo_start, _pbo_end, _pbo_states, pbo_stats_full, _pbo_cmp = _prepare_pbo_data_for_model(
                model_name,
                df_k,
                poster_min=poster_min,
                poster_max=poster_max,
                state_space=state_space,
                grid_k=grid_k,
            )
            pbo_stats = {
                "pbo_comparison_type": str(pbo_stats_full.get("pbo_comparison_type", _pbo_duel_label(model_name))),
                "n_pbo_duels": int(pbo_stats_full["n_pbo_duels"]),
                "n_ties_dropped": int(pbo_stats_full["n_ties_dropped"]),
                "n_start_end_ties": int(pbo_stats_full.get("n_start_end_ties", 0)),
                "n_transition_ties": int(pbo_stats_full.get("n_transition_ties", 0)),
                "n_transition_candidates": int(pbo_stats_full.get("n_transition_candidates", 0)),
                "tie_fraction": float(pbo_stats_full["tie_fraction"]),
            }

        soft_copeland = summarize_soft_copeland_from_idata(
            idata_k,
            var_name="U",
            hdi_prob=float(hdi_prob),
        )
        posterior_pairwise = None
        if U_true_for_metrics is not None:
            posterior_pairwise = compute_posterior_pairwise_accuracy_from_idata(
                idata_k,
                U_true=U_true_for_metrics,
                var_name="U",
                eps_true=1e-6,
                eps_pred=0.0,
            )

        mean = np.asarray(raw_U["mean"], dtype=float)
        sd = np.asarray(raw_U["sd"], dtype=float)
        low = np.asarray(raw_U["low"], dtype=float)
        high = np.asarray(raw_U["high"], dtype=float)

        action_counts = {
            name: int(np.sum(action_idx == idx))
            for idx, name in enumerate(action_order(state_space))
        }

        prefix_rec = {
            "k": int(prefix_idx),
            "k_traj": int(k_traj),
            "is_full_prefix": bool(is_full_prefix),
            "reused_main_idata": bool(reuse_main_idata),
            "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
            "traj_chunk_size": int(step),
            "prev_k_traj": int(prev_k),
            "trajectory_order_provided": trajectory_order is not None,
            "trajectory_permutation_seed": (
                int(trajectory_permutation_seed)
                if trajectory_permutation_seed is not None
                else None
            ),
            "new_traj_start_posters": new_start_posters,
            "model_name": str(model_name),
            "alpha_mode": "inferred" if _infers_alpha_model(model_name) else "fixed",
            "alpha_prior_median": float(alpha_prior_median),
            "alpha_prior_log_sd": float(alpha_prior_log_sd),
            "state_space": str(state_space),
            "grid_k": None if grid_k is None else int(grid_k),
            "action_order": list(action_order(state_space)),
            "preferred_contrast": (
                None if preferred_contrast is None else float(preferred_contrast)
            ),
            "mean": mean,
            "sd": sd,
            "low": low,
            "high": high,
            "n_obs": int(pbo_stats["n_pbo_duels"])
            if _is_pbo_model_name(model_name)
            else int(len(df_k)),
            "n_action_obs": int(len(df_k)),
            "n_traj": int(k_traj),
            "action_counts": action_counts,
            "unique_posters_observed": int(len(np.unique(df_k["poster"].astype(int).to_numpy()))),
            **pbo_stats,
            "mean_norm_draws": np.asarray(norm_U["mean"], dtype=float),
            "sd_norm_draws": np.asarray(norm_U["sd"], dtype=float),
            "low_norm_draws": np.asarray(norm_U["low"], dtype=float),
            "high_norm_draws": np.asarray(norm_U["high"], dtype=float),
            "normalization_norm_draws": norm_U["normalization"],
            "birl_export_mode": str(birl_export_mode),
            "birl_reward_scale": str(birl_reward_scale),
            "birl_value_iters": int(birl_value_iters),
            **({"traj_ids": keep_ids} if store_traj_ids else {}),
        }

        if alpha_summary is not None:
            prefix_rec["alpha_summary"] = alpha_summary

        if soft_copeland is not None:
            prefix_rec.update(
                {
                    "soft_copeland_mean": np.asarray(soft_copeland["mean"], dtype=float),
                    "soft_copeland_sd": np.asarray(soft_copeland["sd"], dtype=float),
                    "soft_copeland_low": np.asarray(soft_copeland["low"], dtype=float),
                    "soft_copeland_high": np.asarray(soft_copeland["high"], dtype=float),
                    "soft_copeland_preference_scale": float(soft_copeland["preference_scale"]),
                }
            )

        if posterior_pairwise is not None:
            prefix_rec.update(
                {
                    "posterior_pairwise_acc_comp": float(
                        posterior_pairwise["posterior_pairwise_acc_comp"]
                    ),
                    "posterior_pairwise_acc_strict": float(
                        posterior_pairwise["posterior_pairwise_acc_strict"]
                    ),
                    "posterior_pairwise_num_pairs_comparable": int(
                        posterior_pairwise["num_pairs_comparable"]
                    ),
                    "posterior_pairwise_expected_correct_comparable": float(
                        posterior_pairwise["posterior_pairwise_expected_correct_comparable"]
                    ),
                    "posterior_pairwise_num_draws": int(posterior_pairwise["n_draws"]),
                    "posterior_pairwise_eps_true": float(posterior_pairwise["eps_true"]),
                    "posterior_pairwise_eps_pred": float(posterior_pairwise["eps_pred"]),
                    "posterior_pairwise_acc_draw_sd": float(
                        posterior_pairwise["posterior_pairwise_acc_draw_sd"]
                    ),
                }
            )

        if bool(save_u_draw_archives) and posterior_archive_dir is not None:
            if bool(reuse_main_idata) and main_u_archive_path:
                prefix_rec.update(
                    {
                        "posterior_archive_u_draws_path": str(main_u_archive_path),
                        "posterior_archive_reuses_main": True,
                    }
                )
            else:
                archive_path = (
                    posterior_archive_dir
                    / "prefix_U_draws"
                    / f"prefix_{int(k_traj):04d}_U_draws.npz"
                )
                archive_info = save_u_draw_archive(
                    idata=idata_k,
                    out_path=archive_path,
                    posters=posters_for_archive,
                    U_true=(
                        U_true_for_metrics
                        if U_true_for_metrics is not None
                        else np.full(num_states, np.nan, dtype=float)
                    ),
                    metadata={
                        **(posterior_archive_metadata or {}),
                        "scope": "prefix",
                        "k": int(prefix_idx),
                        "k_traj": int(k_traj),
                        "is_full_prefix": bool(is_full_prefix),
                        "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                        "traj_ids": keep_ids if store_traj_ids else None,
                    },
                    dtype=str(archive_dtype),
                )
                prefix_rec.update(
                    {
                        "posterior_archive_u_draws_path": str(archive_info["path"]),
                        "posterior_archive_reuses_main": False,
                        "posterior_archive_shape": archive_info["shape"],
                        "posterior_archive_dtype": archive_info["dtype"],
                        "posterior_archive_U_sha256": archive_info.get("U_sha256"),
                        "posterior_archive_posters_sha256": archive_info.get("posters_sha256"),
                        "posterior_archive_U_true_sha256": archive_info.get("U_true_sha256"),
                        "posterior_archive_file_sha256": archive_info.get("archive_file_sha256"),
                    }
                )

        if (
            bool(save_alpha_draw_archives)
            and posterior_archive_dir is not None
            and alpha_summary is not None
        ):
            if bool(reuse_main_idata) and main_alpha_archive_path:
                prefix_rec.update(
                    {
                        "posterior_archive_alpha_draws_path": str(main_alpha_archive_path),
                        "posterior_archive_alpha_reuses_main": True,
                    }
                )
            else:
                alpha_archive_path = (
                    posterior_archive_dir
                    / "alpha_draws"
                    / f"prefix_{int(k_traj):04d}_alpha_draws.npz"
                )
                alpha_archive_info = save_alpha_draw_archive(
                    idata=idata_k,
                    out_path=alpha_archive_path,
                    metadata={
                        **(posterior_archive_metadata or {}),
                        "scope": "prefix",
                        "k": int(prefix_idx),
                        "k_traj": int(k_traj),
                        "is_full_prefix": bool(is_full_prefix),
                        "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                        "traj_ids": keep_ids if store_traj_ids else None,
                    },
                    dtype=str(archive_dtype),
                )
                prefix_rec.update(
                    {
                        "posterior_archive_alpha_draws_path": str(alpha_archive_info["path"]),
                        "posterior_archive_alpha_reuses_main": False,
                        "posterior_archive_alpha_shape": alpha_archive_info["shape"],
                        "posterior_archive_alpha_dtype": alpha_archive_info["dtype"],
                        "posterior_archive_alpha_sha256": alpha_archive_info.get("alpha_sha256"),
                        "posterior_archive_log_alpha_sha256": alpha_archive_info.get("log_alpha_sha256"),
                        "posterior_archive_alpha_file_sha256": alpha_archive_info.get("archive_file_sha256"),
                    }
                )

        if bool(save_full_idata_archive) and posterior_archive_dir is not None:
            if bool(reuse_main_idata) and main_idata_archive_path:
                prefix_rec.update(
                    {
                        "posterior_archive_idata_path": str(main_idata_archive_path),
                        "posterior_archive_idata_reuses_main": True,
                    }
                )
            else:
                idata_path = (
                    posterior_archive_dir
                    / "prefix_idata"
                    / f"prefix_{int(k_traj):04d}_idata.nc"
                )
                idata_path.parent.mkdir(parents=True, exist_ok=True)
                idata_k.to_netcdf(idata_path)
                prefix_rec.update(
                    {
                        "posterior_archive_idata_path": str(idata_path),
                        "posterior_archive_idata_reuses_main": False,
                    }
                )

        if bool(save_diagnostics):
            try:
                diag, per_state_diag = summarize_idata_diagnostics(
                    idata_k,
                    var_name="U",
                    posters=posters_for_archive,
                )
                diag = {
                    **diag,
                    "k": int(prefix_idx),
                    "k_traj": int(k_traj),
                    "is_full_prefix": bool(is_full_prefix),
                    "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                }
            except Exception as e:
                per_state_diag = []
                diag = {
                    "k": int(prefix_idx),
                    "k_traj": int(k_traj),
                    "is_full_prefix": bool(is_full_prefix),
                    "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                    "diagnostics_error": str(e),
                }
            prefix_diag_rows.append(diag)
            prefix_rec["diagnostics"] = diag
            if posterior_archive_dir is not None and per_state_diag:
                diag_dir = posterior_archive_dir / "diagnostics"
                per_state_path = (
                    diag_dir
                    / "prefix_U_diagnostics"
                    / f"prefix_{int(k_traj):04d}_U_diagnostics.csv"
                )
                write_rows_csv(per_state_diag, per_state_path)
                prefix_rec["diagnostics_per_state_csv"] = str(per_state_path)

            if alpha_summary is not None:
                try:
                    alpha_diag, _alpha_rows = summarize_idata_diagnostics(
                        idata_k,
                        var_name="alpha",
                    )
                    alpha_diag = {
                        **alpha_diag,
                        "k": int(prefix_idx),
                        "k_traj": int(k_traj),
                        "is_full_prefix": bool(is_full_prefix),
                        "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                    }
                except Exception as e:
                    alpha_diag = {
                        "var_name": "alpha",
                        "k": int(prefix_idx),
                        "k_traj": int(k_traj),
                        "is_full_prefix": bool(is_full_prefix),
                        "fit_source": "main_idata" if reuse_main_idata else "prefix_mcmc",
                        "diagnostics_error": str(e),
                    }
                prefix_alpha_diag_rows.append(alpha_diag)
                prefix_rec["alpha_diagnostics"] = alpha_diag

        if eval_seen_df is not None and not _is_pbo_model_name(model_name) and state_space == "1d":
            n_eval_traj = int(eval_seen_test_traj_chunk_size) * int(prefix_idx)
            df_eval_k = first_n_trajectories(eval_seen_df, n_traj=n_eval_traj)

            res = compute_elpd_for_df(
                idata=idata_k,
                df_eval=df_eval_k,
                dataset_name=f"seen_test_prefix_{prefix_idx}",
                poster_min=int(poster_min),
                poster_max=int(poster_max),
                model_name=str(model_name),
                gamma_fixed=float(gamma_fixed),
                beta_fixed=float(beta_fixed),
                alpha_fixed=float(alpha_fixed),
                alpha_prior_median=float(alpha_prior_median),
                alpha_prior_log_sd=float(alpha_prior_log_sd),
                likelihood=str(likelihood),
                prior=str(prior),
                gp_eta_sd=float(gp_eta_sd),
                gp_ls_sd=float(gp_ls_sd),
                obs_var_name="obs",
            )

            prefix_rec["elpd_seen_test"] = res.summary
            prefix_rec["elpd_seen_test_per_traj"] = res.per_traj.to_dict(orient="records")
            prefix_rec["elpd_seen_test_by_start"] = res.by_start.to_dict(orient="records")

        prefix_summaries.append(prefix_rec)

    if bool(save_diagnostics) and posterior_archive_dir is not None:
        write_rows_csv(prefix_diag_rows, posterior_archive_dir / "diagnostics" / "prefix_diagnostics.csv")
        if prefix_alpha_diag_rows:
            write_rows_csv(
                prefix_alpha_diag_rows,
                posterior_archive_dir / "diagnostics" / "prefix_alpha_diagnostics.csv",
            )

    return prefix_summaries



def get_or_fit_prefix_summaries(
    *,
    df,
    run_dir: Path,
    poster_min: int,
    poster_max: int,
    gamma_fixed: float,
    beta_fixed: float,
    alpha_fixed: float,
    model_name: str,
    alpha_prior_median: float = 5.0,
    alpha_prior_log_sd: float = 0.75,
    likelihood: str = "argmax",
    prior: str = "rw",
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    cores: int = 1,
    target_accept: float = 0.95,
    random_seed: int = 42,
    cache: bool = True,
    overwrite_cache: bool = False,
    traj_schedule: str = "geom2",
    geom_start: int = 8,
    geom_ratio: int = 2,
    k_traj_list: list[int] | None = None,
    traj_chunk_size: int = 300,
    hdi_prob: float = 0.94,
    eval_seen_df: pd.DataFrame | None = None,
    eval_seen_test_traj_chunk_size: int = 30,
    main_idata=None,
    reuse_main_idata_for_full_prefix: bool = False,
    trajectory_order=None,
    trajectory_permutation_seed: int | None = None,
    U_true_for_metrics=None,
    posters_for_archive=None,
    posterior_archive_dir: str | Path | None = None,
    posterior_archive_metadata: dict | None = None,
    archive_dtype: str = "float32",
    save_u_draw_archives: bool = False,
    save_alpha_draw_archives: bool = False,
    save_diagnostics: bool = False,
    main_u_archive_path: str | None = None,
    main_alpha_archive_path: str | None = None,
    save_full_idata_archive: bool = False,
    main_idata_archive_path: str | None = None,
    birl_export_mode: str = "stay",
    birl_reward_scale: str = "discounted",
    birl_value_iters: int = 1000,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
):
    state_space = normalize_state_space(state_space)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(run_dir)

    raw_ids = df["traj_id"].unique().tolist()
    traj_ids = _resolve_trajectory_order(
        raw_ids,
        trajectory_order=trajectory_order,
    )

    eval_seen_traj_ids = None
    if eval_seen_df is not None and "traj_id" in eval_seen_df.columns:
        raw_eval_ids = eval_seen_df["traj_id"].unique().tolist()
        try:
            eval_seen_traj_ids = sorted(raw_eval_ids, key=lambda x: int(x))
        except Exception:
            eval_seen_traj_ids = sorted(raw_eval_ids)

    prefix_k_traj_list = build_prefix_k_traj_list(
        n_traj=len(traj_ids),
        schedule=str(traj_schedule),
        traj_chunk_size=int(traj_chunk_size),
        geom_start=int(geom_start),
        geom_ratio=int(geom_ratio),
        k_traj_list=k_traj_list,
    )

    key = _cache_key(
        traj_ids=traj_ids,
        poster_min=poster_min,
        poster_max=poster_max,
        gamma_fixed=gamma_fixed,
        beta_fixed=beta_fixed,
        alpha_fixed=alpha_fixed,
        alpha_prior_median=alpha_prior_median,
        alpha_prior_log_sd=alpha_prior_log_sd,
        model_name=model_name,
        likelihood=likelihood,
        prior=prior,
        gp_eta_sd=gp_eta_sd,
        gp_ls_sd=gp_ls_sd,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        k_traj_list=prefix_k_traj_list,
        hdi_prob=hdi_prob,
        eval_seen_traj_ids=eval_seen_traj_ids,
        eval_seen_test_traj_chunk_size=eval_seen_test_traj_chunk_size,
        reuse_main_idata_for_full_prefix=bool(reuse_main_idata_for_full_prefix),
        trajectory_order=traj_ids if trajectory_order is not None else None,
        trajectory_permutation_seed=trajectory_permutation_seed,
        U_true_for_metrics=U_true_for_metrics,
        archive_dtype=archive_dtype,
        save_u_draw_archives=save_u_draw_archives,
        save_alpha_draw_archives=save_alpha_draw_archives,
        save_diagnostics=save_diagnostics,
        main_alpha_archive_path=main_alpha_archive_path,
        save_full_idata_archive=save_full_idata_archive,
        birl_export_mode=str(birl_export_mode),
        birl_reward_scale=str(birl_reward_scale),
        birl_value_iters=int(birl_value_iters),
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )

    if cache and path.exists() and not overwrite_cache:
        cached_key, cached = load_prefix_summaries(path)
        if cached_key == key:
            return cached
        
    if bool(reuse_main_idata_for_full_prefix) and main_idata is None:
            raise ValueError(
                "reuse_main_idata_for_full_prefix=True but main_idata=None. "
                "This is only valid if the caller passes the main full-data idata."
            )
        
    fitted = fit_prefixes_by_trajectory(
        df=df,
        poster_min=poster_min,
        poster_max=poster_max,
        gamma_fixed=gamma_fixed,
        beta_fixed=beta_fixed,
        alpha_fixed=alpha_fixed,
        model_name=model_name,
        alpha_prior_median=alpha_prior_median,
        alpha_prior_log_sd=alpha_prior_log_sd,
        likelihood=likelihood,
        prior=prior,
        gp_eta_sd=gp_eta_sd,
        gp_ls_sd=gp_ls_sd,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        traj_chunk_size=traj_chunk_size,
        hdi_prob=hdi_prob,
        eval_seen_df=eval_seen_df,
        eval_seen_test_traj_chunk_size=int(eval_seen_test_traj_chunk_size),
        k_traj_list=prefix_k_traj_list,
        main_idata=main_idata,
        reuse_main_idata_for_full_prefix=bool(reuse_main_idata_for_full_prefix),
        trajectory_order=traj_ids if trajectory_order is not None else None,
        trajectory_permutation_seed=trajectory_permutation_seed,
        U_true_for_metrics=U_true_for_metrics,
        posters_for_archive=posters_for_archive,
        posterior_archive_dir=posterior_archive_dir,
        posterior_archive_metadata=posterior_archive_metadata,
        archive_dtype=archive_dtype,
        save_u_draw_archives=save_u_draw_archives,
        save_alpha_draw_archives=save_alpha_draw_archives,
        save_diagnostics=save_diagnostics,
        main_u_archive_path=main_u_archive_path,
        main_alpha_archive_path=main_alpha_archive_path,
        save_full_idata_archive=save_full_idata_archive,
        main_idata_archive_path=main_idata_archive_path,
        birl_export_mode=str(birl_export_mode),
        birl_reward_scale=str(birl_reward_scale),
        birl_value_iters=int(birl_value_iters),
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )
    if cache:
        save_prefix_summaries(fitted, path, cache_key=key)
    return fitted
