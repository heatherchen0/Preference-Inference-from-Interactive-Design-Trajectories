# src/elpd.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pymc as pm
from .preprocess import prepare_inverse_data
from .model import (
    build_model2_discounted,
    build_model4_endpoint_discounted_boltzmann,
    build_model4_endpoint_discounted_boltzmann_infer_alpha,
)


def _logmeanexp(a: np.ndarray, axis: int = 0, eps: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.mean(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass(frozen=True)
class ELPDResult:
    summary: dict
    per_traj: pd.DataFrame
    by_start: pd.DataFrame


def _extract_ll_array(ll_obj: Any, obs_var_name: str = "obs") -> np.ndarray:
    if hasattr(ll_obj, "log_likelihood"):
        ll_da = ll_obj.log_likelihood[obs_var_name]
        return np.asarray(ll_da.values, dtype=float)

    try:
        ll_da = ll_obj[obs_var_name]
        return np.asarray(getattr(ll_da, "values", ll_da), dtype=float)
    except Exception as e:
        raise TypeError(
            "Could not extract log-likelihood array from pm.compute_log_likelihood output. "
            f"type={type(ll_obj)} keys={getattr(ll_obj, 'keys', lambda: None)()}"
        ) from e


def _build_eval_model(
    *,
    model_name: str,
    state_idx: np.ndarray,
    action_idx: np.ndarray,
    num_states: int,
    gamma_fixed: float,
    beta_fixed: float,
    alpha_fixed: float,
    alpha_prior_median: float = 5.0,
    alpha_prior_log_sd: float = 0.75,
    likelihood: str = "argmax",
    prior: str = "rw",
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
):
    name = str(model_name).lower()
    if name == "model2_discounted":
        return build_model2_discounted(
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

    if name == "model4_endpoint_discounted_boltzmann":
        return build_model4_endpoint_discounted_boltzmann(
            state_idx=state_idx,
            action_idx=action_idx,
            num_states=num_states,
            gamma_fixed=float(gamma_fixed),
            alpha_fixed=float(alpha_fixed),
            prior=str(prior).lower(),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
        )

    if name == "model4_endpoint_discounted_boltzmann_infer_alpha":
        return build_model4_endpoint_discounted_boltzmann_infer_alpha(
            state_idx=state_idx,
            action_idx=action_idx,
            num_states=num_states,
            gamma_fixed=float(gamma_fixed),
            alpha_prior_median=float(alpha_prior_median),
            alpha_prior_log_sd=float(alpha_prior_log_sd),
            prior=str(prior).lower(),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
        )

    raise ValueError(f"Unknown model_name={model_name!r}")


def compute_elpd_for_df(
    *,
    idata,
    df_eval: pd.DataFrame,
    dataset_name: str,
    poster_min: int,
    poster_max: int,
    model_name: str,
    gamma_fixed: float,
    beta_fixed: float,
    alpha_fixed: float,
    alpha_prior_median: float = 5.0,
    alpha_prior_log_sd: float = 0.75,
    likelihood: str = "argmax",
    prior: str = "rw",
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
    obs_var_name: str = "obs",
) -> ELPDResult:
    required = {"poster", "action", "traj_id", "t"}
    missing = required - set(df_eval.columns)
    if missing:
        raise ValueError(f"compute_elpd_for_df: missing columns: {missing}")

    df = df_eval.sort_values(["traj_id", "t"]).reset_index(drop=True)

    state_idx, action_idx, num_states = prepare_inverse_data(
        df, poster_min=int(poster_min), poster_max=int(poster_max)
    )

    eval_model = _build_eval_model(
        model_name=model_name,
        state_idx=state_idx,
        action_idx=action_idx,
        num_states=int(num_states),
        gamma_fixed=float(gamma_fixed),
        beta_fixed=float(beta_fixed),
        alpha_fixed=float(alpha_fixed),
        alpha_prior_median=float(alpha_prior_median),
        alpha_prior_log_sd=float(alpha_prior_log_sd),
        likelihood=str(likelihood),
        prior=str(prior),
        gp_eta_sd=float(gp_eta_sd),
        gp_ls_sd=float(gp_ls_sd),
    )

    ll_obj = pm.compute_log_likelihood(
        idata,
        model=eval_model,
        extend_inferencedata=False,
        var_names=[str(obs_var_name)],
    )
    ll = _extract_ll_array(ll_obj, obs_var_name=str(obs_var_name))

    chains, draws, n_obs = ll.shape
    n_samples = int(chains * draws)
    ll_samp = ll.reshape(n_samples, n_obs)

    traj_ids = df["traj_id"].to_numpy()
    posters = df["poster"].astype(int).to_numpy()

    if n_obs == 0:
        raise ValueError("compute_elpd_for_df: eval dataframe is empty.")

    change = np.flatnonzero(traj_ids[1:] != traj_ids[:-1])
    starts = np.r_[0, change + 1]
    traj_ids_u = traj_ids[starts]
    start_posters = posters[starts]
    counts = np.diff(np.r_[starts, n_obs]).astype(int)

    ll_samp_traj = np.add.reduceat(ll_samp, starts, axis=1)

    elpd_traj = _logmeanexp(ll_samp_traj, axis=0)
    elpd_traj = np.asarray(elpd_traj, dtype=float).reshape(-1)

    per_traj = pd.DataFrame(
        {
            "dataset": str(dataset_name),
            "traj_id": traj_ids_u,
            "start_poster": start_posters.astype(int),
            "n_steps": counts,
            "elpd_traj": elpd_traj,
            "elpd_per_step_traj": elpd_traj / np.maximum(counts, 1),
        }
    )

    elpd_total = float(np.sum(elpd_traj))
    n_traj = int(per_traj.shape[0])
    n_steps_total = int(np.sum(counts))

    by_start_rows = []
    for sp in sorted(per_traj["start_poster"].unique().tolist()):
        d = per_traj[per_traj["start_poster"] == sp]
        by_start_rows.append(
            {
                "dataset": str(dataset_name),
                "start_poster": int(sp),
                "n_traj": int(len(d)),
                "n_steps": int(d["n_steps"].sum()),
                "elpd_total": float(d["elpd_traj"].sum()),
                "elpd_per_traj": float(d["elpd_traj"].mean()) if len(d) else np.nan,
                "elpd_per_step": float(d["elpd_traj"].sum() / max(int(d["n_steps"].sum()), 1)),
            }
        )
    by_start = pd.DataFrame(by_start_rows)

    summary = {
        "dataset": str(dataset_name),
        "n_samples": int(n_samples),
        "n_obs": int(n_obs),
        "n_traj": int(n_traj),
        "n_steps": int(n_steps_total),
        "elpd_total": float(elpd_total),
        "elpd_per_traj": float(elpd_total / max(n_traj, 1)),
        "elpd_per_step": float(elpd_total / max(n_steps_total, 1)),
    }

    return ELPDResult(summary=summary, per_traj=per_traj, by_start=by_start)


def first_n_trajectories(df: pd.DataFrame, n_traj: int) -> pd.DataFrame:
    if int(n_traj) <= 0:
        return df.iloc[0:0].copy()

    d = df.sort_values(["traj_id", "t"]).reset_index(drop=True)
    traj_ids = d["traj_id"].unique().tolist()
    keep = set(traj_ids[: int(n_traj)])
    return d[d["traj_id"].isin(keep)].copy()
