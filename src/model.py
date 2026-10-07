import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .state_space import (
    ACTION_ORDER_2D,
    infer_grid_k,
    manhattan_distance_matrix_2d,
    neighbor_indices_and_mask_2d,
)


def softmax_stable(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def model2_action_probs_numpy(U, state_idx, gamma, beta):
    U = np.asarray(U, dtype=float)
    state_idx = np.asarray(state_idx, dtype=np.int64)

    S = U.shape[0]
    h = np.arange(S)

    distances = np.abs(h[None, :] - state_idx[:, None])
    scores = beta * (gamma ** distances) * U[None, :]
    target_probs = softmax_stable(scores, axis=1)

    dec_mask = h[None, :] < state_idx[:, None]
    export_mask = h[None, :] == state_idx[:, None]
    inc_mask = h[None, :] > state_idx[:, None]

    p_dec = np.sum(target_probs * dec_mask, axis=1)
    p_export = np.sum(target_probs * export_mask, axis=1)
    p_inc = np.sum(target_probs * inc_mask, axis=1)

    return np.column_stack([p_dec, p_export, p_inc])


def categorical_loglik(action_probs, action_idx, eps=1e-12):
    p_obs = action_probs[np.arange(len(action_idx)), action_idx]
    return np.sum(np.log(p_obs + eps))


def _gp_inputs_for_prior(*, S: int, gp_inputs=None) -> np.ndarray:
    if gp_inputs is None:
        if int(S) <= 1:
            raise ValueError("num_states must be >= 2 for GP input rescaling.")
        return np.linspace(0.0, 1.0, int(S), dtype=np.float64)[:, None]

    x = np.asarray(gp_inputs, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"gp_inputs must be a 2D array, got shape {x.shape}.")
    if x.shape[0] != int(S):
        raise ValueError(f"gp_inputs has {x.shape[0]} rows but expected {S}.")
    return x


def _build_bounded_utility_prior(
    *,
    S: int,
    prior: str,
    mean_logit_sd: float = 1.0,
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
    gp_inputs=None,
):
    prior = str(prior).strip().lower()

    if prior in ("iid", "iid_uniform", "uniform", "beta_uniform"):
        return pm.Beta("U", alpha=1.0, beta=1.0, shape=int(S))

    if prior == "rw":
        rw_sigma = pm.HalfNormal("rw_sigma", sigma=1.0)
        f_raw = pm.GaussianRandomWalk(
            "f_raw",
            sigma=rw_sigma,
            init_dist=pm.Normal.dist(mu=0.0, sigma=1.0),
            shape=int(S),
        )
        pm.Potential(
            "soft_center_mean_logit",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=float(mean_logit_sd)), pt.mean(f_raw)),
        )
        return pm.Deterministic("U", pm.math.sigmoid(f_raw))

    if prior in ("gp", "gp_rbf", "rbf"):
        x = _gp_inputs_for_prior(S=int(S), gp_inputs=gp_inputs)
        X = pm.Data("gp_X", x)

        gp_eta = pm.HalfNormal("gp_eta", sigma=float(gp_eta_sd))
        gp_ls = pm.HalfNormal("gp_ls", sigma=float(gp_ls_sd))

        cov = (gp_eta**2) * pm.gp.cov.ExpQuad(input_dim=int(x.shape[1]), ls=gp_ls)
        gp = pm.gp.Latent(cov_func=cov)
        f_raw = gp.prior("f_raw", X=X)

        pm.Potential(
            "soft_center_mean_logit",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=float(mean_logit_sd)), pt.mean(f_raw)),
        )
        return pm.Deterministic("U", pm.math.sigmoid(f_raw))

    raise ValueError(f"Unknown prior={prior!r} (use 'iid', 'rw', or 'gp_rbf').")


def build_model2_discounted(
    state_idx,
    action_idx,
    num_states,
    gamma_fixed=0.9,
    beta_fixed=20.0,
    likelihood="argmax",          # "softmax" or "argmax"
    prior="rw",                   # "rw" or "gp_rbf"
    eps_fixed=0.01,
    infer_eps=False,
    mean_logit_sd=1.0,            # larger => weaker soft-centering
    tie_jitter=1e-6,
    # GP hyperpriors
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
):
    """
    Inference model for User behavior model 2: goal-directed with distance discounting.
    """
    with pm.Model() as smooth_model:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)

        h = pt.arange(num_states)
        distances = pt.abs(h[None, :] - state_data[:, None])

        prior = str(prior).lower()
        if prior == "rw":
            rw_sigma = pm.HalfNormal("rw_sigma", sigma=1.0)
            f_raw = pm.GaussianRandomWalk(
                "f_raw",
                sigma=rw_sigma,
                init_dist=pm.Normal.dist(mu=0.0, sigma=1.0),
                shape=num_states,
            )

        elif prior in ("gp", "gp_rbf", "rbf"):
            if num_states <= 1:
                raise ValueError("num_states must be >= 2 for GP input rescaling.")

            x = np.linspace(0.0, 1.0, num_states, dtype=np.float64)[:, None]
            X = pm.Data("gp_X", x)

            gp_eta = pm.HalfNormal("gp_eta", sigma=float(gp_eta_sd))
            gp_ls = pm.HalfNormal("gp_ls", sigma=float(gp_ls_sd))

            cov = (gp_eta**2) * pm.gp.cov.ExpQuad(input_dim=1, ls=gp_ls)
            gp = pm.gp.Latent(cov_func=cov)

            f_raw = gp.prior("f_raw", X=X)

        else:
            raise ValueError(f"Unknown prior={prior!r} (use 'rw' or 'gp_rbf').")

        if likelihood == "softmax":
            f_centered = f_raw - pt.mean(f_raw)
            pm.Deterministic("f_centered", f_centered)

            U = pm.Deterministic("U", pm.math.sigmoid(f_centered))

            scores = beta_fixed * (gamma_fixed ** distances) * U[None, :]
            target_probs = pm.math.softmax(scores, axis=1)

            dec_mask = pt.cast(pt.lt(h[None, :], state_data[:, None]), "float64")
            export_mask = pt.cast(pt.eq(h[None, :], state_data[:, None]), "float64")
            inc_mask = pt.cast(pt.gt(h[None, :], state_data[:, None]), "float64")

            p_dec = pt.sum(target_probs * dec_mask, axis=1)
            p_export = pt.sum(target_probs * export_mask, axis=1)
            p_inc = pt.sum(target_probs * inc_mask, axis=1)

            action_probs = pt.stack([p_dec, p_export, p_inc], axis=1)

        elif likelihood == "argmax":
            U = pm.Deterministic("U", pm.math.sigmoid(f_raw))

            pm.Potential(
                "soft_center_mean_logit",
                pm.logp(pm.Normal.dist(mu=0.0, sigma=mean_logit_sd), pt.mean(f_raw)),
            )

            discounted_value = (gamma_fixed ** distances) * U[None, :]

            #jitter_vec = (-float(tie_jitter)) * h
            #h_star = pt.argmax(discounted_value + jitter_vec[None, :], axis=1)
            h_star = pt.argmax(discounted_value, axis=1)
            
            implied_action = pt.switch(
                pt.lt(h_star, state_data),
                0,
                pt.switch(pt.gt(h_star, state_data), 2, 1),
            )

            pm.Deterministic("h_star", h_star)
            pm.Deterministic("implied_action", implied_action)

            if infer_eps:
                eps = pm.Beta("eps", alpha=2, beta=50)
            else:
                eps = pt.as_tensor_variable(float(eps_fixed))

            p_correct = 1.0 - eps
            p_wrong = eps / 2.0

            action_probs = pt.ones((state_data.shape[0], 3), dtype="float64") * p_wrong
            action_probs = pt.set_subtensor(
                action_probs[pt.arange(state_data.shape[0]), implied_action],
                p_correct,
            )

        else:
            raise ValueError(f"Unknown likelihood={likelihood!r}")

        pm.Deterministic("action_probs", action_probs)
        pm.Categorical("obs", p=action_probs, observed=action_data)

    return smooth_model


def build_model4_endpoint_discounted_boltzmann(
    state_idx,
    action_idx,
    num_states,
    gamma_fixed=0.995,
    alpha_fixed=5.0,
    prior="rw",               # "iid", "rw", "gp_rbf"
    mean_logit_sd=1.0,
    # GP hyperpriors
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    """
    User behavior model 4: endpoint-only utility with discounting and Boltzmann rationality.
    """
    S = int(num_states)
    state_idx = np.asarray(state_idx, dtype=np.int64)
    action_idx = np.asarray(action_idx, dtype=np.int64)

    if not (0.0 < float(gamma_fixed) < 1.0):
        raise ValueError(f"gamma_fixed must be in (0,1), got {gamma_fixed}")

    with pm.Model() as m:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)

        U = _build_bounded_utility_prior(
            S=S,
            prior=str(prior).lower(),
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        gamma = pt.as_tensor_variable(float(gamma_fixed))
        alpha = pt.as_tensor_variable(float(alpha_fixed))

        h = pt.arange(S)
        D = pt.abs(h[:, None] - h[None, :])
        disc = gamma ** D
        V = pt.max(disc * U[None, :], axis=1)
        pm.Deterministic("V", V)

        neg_inf = pt.constant(-np.inf, dtype="float64")

        Q_dec = pt.concatenate([pt.stack([neg_inf]), gamma * V[:-1]])
        Q_export = U
        Q_inc = pt.concatenate([gamma * V[1:], pt.stack([neg_inf])])

        Q_all = pt.stack([Q_dec, Q_export, Q_inc], axis=1)
        pm.Deterministic("Q_all", Q_all)

        Q_obs = Q_all[state_data]  # (N,3)
        action_probs = pm.Deterministic(
            "action_probs",
            pm.math.softmax(alpha * Q_obs, axis=1),
        )

        pm.Categorical("obs", p=action_probs, observed=action_data)

    return m


def build_model4_endpoint_discounted_boltzmann_infer_alpha(
    state_idx,
    action_idx,
    num_states,
    gamma_fixed=0.995,
    alpha_prior_median=5.0,
    alpha_prior_log_sd=0.75,
    prior="rw",               # "iid", "rw", "gp_rbf"
    mean_logit_sd=1.0,
    # GP hyperpriors
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    """
    1D endpoint-only Model 4 with a globally inferred Boltzmann temperature.
    """
    S = int(num_states)
    state_idx = np.asarray(state_idx, dtype=np.int64)
    action_idx = np.asarray(action_idx, dtype=np.int64)

    if not (0.0 < float(gamma_fixed) < 1.0):
        raise ValueError(f"gamma_fixed must be in (0,1), got {gamma_fixed}")
    if float(alpha_prior_median) <= 0.0:
        raise ValueError(
            f"alpha_prior_median must be positive, got {alpha_prior_median}"
        )
    if float(alpha_prior_log_sd) <= 0.0:
        raise ValueError(
            f"alpha_prior_log_sd must be positive, got {alpha_prior_log_sd}"
        )

    with pm.Model() as m:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)

        U = _build_bounded_utility_prior(
            S=S,
            prior=str(prior).lower(),
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        gamma = pt.as_tensor_variable(float(gamma_fixed))
        alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0)
        log_alpha = pm.Deterministic(
            "log_alpha",
            float(np.log(float(alpha_prior_median)))
            + float(alpha_prior_log_sd) * alpha_raw,
        )
        alpha = pm.Deterministic("alpha", pt.exp(log_alpha))

        h = pt.arange(S)
        D = pt.abs(h[:, None] - h[None, :])
        disc = gamma ** D
        V = pt.max(disc * U[None, :], axis=1)
        pm.Deterministic("V", V)

        neg_inf = pt.constant(-np.inf, dtype="float64")

        Q_dec = pt.concatenate([pt.stack([neg_inf]), gamma * V[:-1]])
        Q_export = U
        Q_inc = pt.concatenate([gamma * V[1:], pt.stack([neg_inf])])

        Q_all = pt.stack([Q_dec, Q_export, Q_inc], axis=1)
        pm.Deterministic("Q_all", Q_all)

        Q_obs = Q_all[state_data]  # (N,3)
        valid_q = Q_obs > neg_inf
        Q_obs_safe = pt.switch(valid_q, Q_obs, 0.0)
        logits_obs = pt.switch(valid_q, alpha * Q_obs_safe, neg_inf)
        action_probs = pm.Deterministic(
            "action_probs",
            pm.math.softmax(logits_obs, axis=1),
        )

        pm.Categorical("obs", p=action_probs, observed=action_data)

    return m

### BIRL baseline ###
def _canonical_birl_export_mode(export_mode: str) -> str:
    mode = str(export_mode).strip().lower()
    aliases = {
        "stay": "stay",
        "self_loop": "stay",
        "self-loop": "stay",
        "loop": "stay",

        "terminal": "terminal",
        "term": "terminal",
        "stop": "terminal",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown birl export_mode={export_mode!r}. "
            "Use 'stay' or 'terminal'."
        )
    return aliases[mode]


def _canonical_birl_reward_scale(reward_scale: str) -> str:
    scale = str(reward_scale).strip().lower()
    aliases = {
        "discounted": "discounted",
        "scaled": "discounted",
        "one_minus_gamma": "discounted",
        "1-gamma": "discounted",
        "(1-gamma)": "discounted",

        "raw": "raw",
        "unscaled": "raw",
        "none": "raw",
        "rho": "raw",
    }
    if scale not in aliases:
        raise ValueError(
            f"Unknown birl reward_scale={reward_scale!r}. "
            "Use 'discounted' or 'raw'."
        )
    return aliases[scale]


def _build_birl_rho_prior(
    *,
    S: int,
    prior: str,
    mean_logit_sd: float = 1.0,
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
    gp_inputs=None,
):
    prior = str(prior).strip().lower()

    if prior in ("iid", "iid_uniform", "uniform", "beta_uniform"):
        rho = pm.Beta("U", alpha=1.0, beta=1.0, shape=S)
        return rho

    if prior in ("gp", "gp_rbf", "rbf", "logistic_gp", "logistic_gp_rbf"):
        x = _gp_inputs_for_prior(S=int(S), gp_inputs=gp_inputs)
        X = pm.Data("gp_X", x)

        gp_eta = pm.HalfNormal("gp_eta", sigma=float(gp_eta_sd))
        gp_ls = pm.HalfNormal("gp_ls", sigma=float(gp_ls_sd))

        cov = (gp_eta**2) * pm.gp.cov.ExpQuad(input_dim=int(x.shape[1]), ls=gp_ls)
        gp = pm.gp.Latent(cov_func=cov)
        f_raw = gp.prior("f_raw", X=X)

        pm.Potential(
            "soft_center_mean_logit",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=mean_logit_sd), pt.mean(f_raw)),
        )

        rho = pm.Deterministic("U", pm.math.sigmoid(f_raw))
        return rho

    if prior in (
        "gp_uniform",
        "gp_uniform_rbf",
        "gp_copula",
        "gp_copula_rbf",
        "copula_gp",
        "copula_gp_rbf",
    ):
        x = _gp_inputs_for_prior(S=int(S), gp_inputs=gp_inputs)
        X = pm.Data("gp_X", x)

        gp_ls = pm.HalfNormal("gp_ls", sigma=float(gp_ls_sd))

        # Unit-variance RBF covariance gives z_i marginal N(0,1),
        # so Phi(z_i) is marginal Uniform(0,1).
        cov = pm.gp.cov.ExpQuad(input_dim=int(x.shape[1]), ls=gp_ls)
        gp = pm.gp.Latent(cov_func=cov)
        z_raw = gp.prior("z_raw", X=X)

        rho = pm.Deterministic(
            "U",
            0.5 * (1.0 + pt.erf(z_raw / np.sqrt(2.0))),
        )
        return rho

    raise ValueError(
        f"Unknown BIRL prior={prior!r}. "
        "Use 'iid', 'gp_rbf', or 'gp_uniform_rbf'."
    )


def _birl_q_from_value(
    *,
    R,
    V,
    gamma,
    export_mode: str,
):
    S = R.shape[0]
    neg_inf = pt.constant(-np.inf, dtype="float64")

    if S == 1:
        Q_dec = pt.stack([neg_inf])
        Q_inc = pt.stack([neg_inf])
    else:
        Q_dec = pt.concatenate([
            pt.stack([neg_inf]),
            R[1:] + gamma * V[:-1],
        ])
        Q_inc = pt.concatenate([
            R[:-1] + gamma * V[1:],
            pt.stack([neg_inf]),
        ])

    if export_mode == "stay":
        Q_export = R + gamma * V
    elif export_mode == "terminal":
        Q_export = R
    else:
        raise AssertionError(f"Unhandled export_mode={export_mode!r}")

    Q_all = pt.stack([Q_dec, Q_export, Q_inc], axis=1)
    return Q_all


def _birl_value_iteration(
    *,
    R,
    gamma,
    S: int,
    export_mode: str,
    n_iter: int,
):
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}.")

    V0 = pt.zeros((int(S),), dtype="float64")

    def _step(V_prev, R_, gamma_):
        Q_prev = _birl_q_from_value(
            R=R_,
            V=V_prev,
            gamma=gamma_,
            export_mode=export_mode,
        )
        return pt.max(Q_prev, axis=1)

    V_seq, _updates = pytensor.scan(
        fn=_step,
        outputs_info=V0,
        non_sequences=[R, gamma],
        n_steps=n_iter,
        strict=True,
    )

    return V_seq[-1]


def build_birl_baseline_discounted_boltzmann(
    state_idx,
    action_idx,
    num_states,
    gamma_fixed=0.995,
    alpha_fixed=5.0,
    prior="iid",                  # "iid", "gp_rbf", or "gp_uniform_rbf"
    export_mode="stay",           # "stay" or "terminal"
    reward_scale="discounted",    # "discounted" or "raw"
    value_iters=1000,
    mean_logit_sd=1.0,
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    S = int(num_states)
    state_idx = np.asarray(state_idx, dtype=np.int64)
    action_idx = np.asarray(action_idx, dtype=np.int64)

    gamma_fixed = float(gamma_fixed)
    alpha_fixed = float(alpha_fixed)

    if not (0.0 < gamma_fixed < 1.0):
        raise ValueError(f"gamma_fixed must be in (0,1), got {gamma_fixed}")

    export_mode = _canonical_birl_export_mode(export_mode)
    reward_scale = _canonical_birl_reward_scale(reward_scale)

    with pm.Model() as m:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)

        rho = _build_birl_rho_prior(
            S=S,
            prior=prior,
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        gamma = pt.as_tensor_variable(gamma_fixed)
        alpha = pt.as_tensor_variable(alpha_fixed)

        if reward_scale == "discounted":
            reward_scale_factor = 1.0 - gamma_fixed
        elif reward_scale == "raw":
            reward_scale_factor = 1.0
        else:
            raise AssertionError(f"Unhandled reward_scale={reward_scale!r}")

        pm.Deterministic("reward_scale_factor", pt.as_tensor_variable(float(reward_scale_factor)))

        R = pm.Deterministic("R", float(reward_scale_factor) * rho)

        V = _birl_value_iteration(
            R=R,
            gamma=gamma,
            S=S,
            export_mode=export_mode,
            n_iter=int(value_iters),
        )
        V = pm.Deterministic("V", V)

        Q_all = _birl_q_from_value(
            R=R,
            V=V,
            gamma=gamma,
            export_mode=export_mode,
        )
        Q_all = pm.Deterministic("Q_all", Q_all)

        Q_obs = Q_all[state_data]

        action_probs = pm.Deterministic(
            "action_probs",
            pm.math.softmax(alpha * Q_obs, axis=1),
        )

        pm.Categorical("obs", p=action_probs, observed=action_data)

    return m


def _birl_q_from_value_2d(
    *,
    R,
    V,
    gamma,
    export_mode: str,
    neighbor_idx,
    valid_mask,
):
    neg_inf = pt.constant(-np.inf, dtype="float64")

    Q_all = R[:, None] + gamma * V[neighbor_idx]
    if export_mode == "stay":
        Q_export = R + gamma * V
    elif export_mode == "terminal":
        Q_export = R
    else:
        raise AssertionError(f"Unhandled export_mode={export_mode!r}")

    Q_all = pt.set_subtensor(Q_all[:, 0], Q_export)
    Q_all = pt.switch(valid_mask, Q_all, neg_inf)
    return Q_all


def _birl_value_iteration_2d(
    *,
    R,
    gamma,
    S: int,
    export_mode: str,
    neighbor_idx,
    valid_mask,
    n_iter: int,
):
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}.")

    V0 = pt.zeros((int(S),), dtype="float64")

    def _step(V_prev, R_, gamma_, neighbor_idx_, valid_mask_):
        Q_prev = _birl_q_from_value_2d(
            R=R_,
            V=V_prev,
            gamma=gamma_,
            export_mode=export_mode,
            neighbor_idx=neighbor_idx_,
            valid_mask=valid_mask_,
        )
        return pt.max(Q_prev, axis=1)

    V_seq, _updates = pytensor.scan(
        fn=_step,
        outputs_info=V0,
        non_sequences=[R, gamma, neighbor_idx, valid_mask],
        n_steps=n_iter,
        strict=True,
    )

    return V_seq[-1]


def build_birl_baseline_discounted_boltzmann_2d(
    state_idx,
    action_idx,
    num_states,
    *,
    grid_k: int = 10,
    gamma_fixed=0.95,
    alpha_fixed=5.0,
    prior="gp_rbf",
    export_mode="stay",
    reward_scale="discounted",
    value_iters=100,
    mean_logit_sd=1.0,
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    S = int(num_states)
    k = infer_grid_k(S, grid_k=int(grid_k))
    state_idx = np.asarray(state_idx, dtype=np.int64)
    action_idx = np.asarray(action_idx, dtype=np.int64)

    gamma_fixed = float(gamma_fixed)
    alpha_fixed = float(alpha_fixed)

    if not (0.0 < gamma_fixed < 1.0):
        raise ValueError(f"gamma_fixed must be in (0,1), got {gamma_fixed}")
    if np.any((action_idx < 0) | (action_idx >= len(ACTION_ORDER_2D))):
        raise ValueError("2D action_idx contains out-of-range action ids.")

    export_mode = _canonical_birl_export_mode(export_mode)
    reward_scale = _canonical_birl_reward_scale(reward_scale)
    neighbors, valid_mask = neighbor_indices_and_mask_2d(grid_k=k)

    with pm.Model() as m:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)

        rho = _build_birl_rho_prior(
            S=S,
            prior=prior,
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        gamma = pt.as_tensor_variable(gamma_fixed)
        alpha = pt.as_tensor_variable(alpha_fixed)

        if reward_scale == "discounted":
            reward_scale_factor = 1.0 - gamma_fixed
        elif reward_scale == "raw":
            reward_scale_factor = 1.0
        else:
            raise AssertionError(f"Unhandled reward_scale={reward_scale!r}")

        pm.Deterministic("reward_scale_factor", pt.as_tensor_variable(float(reward_scale_factor)))
        R = pm.Deterministic("R", float(reward_scale_factor) * rho)

        neighbor_idx = pt.as_tensor_variable(neighbors)
        valid = pt.as_tensor_variable(valid_mask)

        V = _birl_value_iteration_2d(
            R=R,
            gamma=gamma,
            S=S,
            export_mode=export_mode,
            neighbor_idx=neighbor_idx,
            valid_mask=valid,
            n_iter=int(value_iters),
        )
        V = pm.Deterministic("V", V)

        Q_all = _birl_q_from_value_2d(
            R=R,
            V=V,
            gamma=gamma,
            export_mode=export_mode,
            neighbor_idx=neighbor_idx,
            valid_mask=valid,
        )
        Q_all = pm.Deterministic("Q_all", Q_all)

        Q_obs = Q_all[state_data]
        action_probs = pm.Deterministic(
            "action_probs",
            pm.math.softmax(alpha * Q_obs, axis=1),
        )
        pm.Categorical("obs", p=action_probs, observed=action_data)

    return m


### PBO baseline ###
def _build_pbo_utility_prior(
    *,
    S: int,
    prior: str,
    mean_logit_sd: float = 1.0,
    gp_eta_sd: float = 1.0,
    gp_ls_sd: float = 0.25,
    gp_inputs=None,
):
    return _build_bounded_utility_prior(
        S=int(S),
        prior=str(prior).strip().lower(),
        mean_logit_sd=float(mean_logit_sd),
        gp_eta_sd=float(gp_eta_sd),
        gp_ls_sd=float(gp_ls_sd),
        gp_inputs=gp_inputs,
    )


def build_pbo_pairwise_preference_model(
    start_idx,
    end_idx,
    num_states,
    prior="gp_rbf",
    preference_scale_fixed=1.0,
    mean_logit_sd=1.0,
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    """
    Offline PBO-style pairwise preference model.

    The caller supplies loser/winner state-index pairs. For start-end PBO this
    means start_idx=start and end_idx=export; for transition PBO this means
    start_idx=current state and end_idx=next state.
    """
    S = int(num_states)
    start_idx = np.asarray(start_idx, dtype=np.int64).reshape(-1)
    end_idx = np.asarray(end_idx, dtype=np.int64).reshape(-1)

    if start_idx.shape != end_idx.shape:
        raise ValueError(
            f"start_idx and end_idx must have the same shape, got "
            f"{start_idx.shape} vs {end_idx.shape}"
        )
    if S <= 0:
        raise ValueError(f"num_states must be positive, got {num_states}")
    if start_idx.size:
        if np.any(start_idx < 0) or np.any(start_idx >= S):
            raise ValueError("start_idx contains out-of-range state indices.")
        if np.any(end_idx < 0) or np.any(end_idx >= S):
            raise ValueError("end_idx contains out-of-range state indices.")

    observed_wins = np.ones(start_idx.shape[0], dtype=np.int64)

    with pm.Model() as m:
        start_data = pm.Data("start_idx", start_idx)
        end_data = pm.Data("end_idx", end_idx)

        U = _build_pbo_utility_prior(
            S=S,
            prior=str(prior),
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        scale = pt.as_tensor_variable(float(preference_scale_fixed))

        if start_idx.size:
            logits = scale * (U[end_data] - U[start_data])
            p_pref = pm.Deterministic("p_pref", pm.math.sigmoid(logits))
            pm.Bernoulli("obs", p=p_pref, observed=observed_wins)
        else:
            pm.Deterministic("p_pref", pt.zeros((0,), dtype="float64"))
            pm.Potential("no_pbo_comparisons", pt.as_tensor_variable(0.0))

    return m


def build_model4_endpoint_discounted_boltzmann_2d(
    state_idx,
    action_idx,
    num_states,
    *,
    grid_k: int = 10,
    gamma_fixed=0.95,
    alpha_fixed=5.0,
    prior="gp_rbf",
    mean_logit_sd=1.0,
    gp_eta_sd=1.0,
    gp_ls_sd=0.25,
    gp_inputs=None,
):
    """
    2D endpoint-only utility model with Manhattan-distance editing geometry.

    Action order is:
        export, inc-headline, dec-headline, inc-background, dec-background
    """
    S = int(num_states)
    k = infer_grid_k(S, grid_k=int(grid_k))
    state_idx = np.asarray(state_idx, dtype=np.int64)
    action_idx = np.asarray(action_idx, dtype=np.int64)

    if not (0.0 < float(gamma_fixed) < 1.0):
        raise ValueError(f"gamma_fixed must be in (0,1), got {gamma_fixed}")
    if np.any((action_idx < 0) | (action_idx >= len(ACTION_ORDER_2D))):
        raise ValueError("2D action_idx contains out-of-range action ids.")

    D = manhattan_distance_matrix_2d(grid_k=k)
    neighbors, valid_mask = neighbor_indices_and_mask_2d(grid_k=k)

    with pm.Model() as m:
        state_data = pm.Data("state_idx", state_idx)
        action_data = pm.Data("action_idx", action_idx)
        D_data = pm.Data("manhattan_distance", D)

        U = _build_bounded_utility_prior(
            S=S,
            prior=str(prior).lower(),
            mean_logit_sd=float(mean_logit_sd),
            gp_eta_sd=float(gp_eta_sd),
            gp_ls_sd=float(gp_ls_sd),
            gp_inputs=gp_inputs,
        )

        gamma = pt.as_tensor_variable(float(gamma_fixed))
        alpha = pt.as_tensor_variable(float(alpha_fixed))

        V = pt.max((gamma ** D_data) * U[None, :], axis=1)
        pm.Deterministic("V", V)

        neighbor_idx = pt.as_tensor_variable(neighbors)
        valid = pt.as_tensor_variable(valid_mask)
        neg_inf = pt.constant(-np.inf, dtype="float64")

        Q_all = gamma * V[neighbor_idx]
        Q_all = pt.set_subtensor(Q_all[:, 0], U)
        Q_all = pt.switch(valid, Q_all, neg_inf)
        Q_all = pm.Deterministic("Q_all", Q_all)

        Q_obs = Q_all[state_data]
        action_probs = pm.Deterministic(
            "action_probs",
            pm.math.softmax(alpha * Q_obs, axis=1),
        )
        pm.Categorical("obs", p=action_probs, observed=action_data)

    return m


# Backward-compatible alias for older scripts/results code. New code should use
# build_pbo_pairwise_preference_model because this builder is shared by both
# start-end PBO and transition-adjacent PBO.
build_pbo_baseline_start_end_preference = build_pbo_pairwise_preference_model
