import numpy as np

from .env import (
    DEC_BACKGROUND,
    DEC_HEADLINE,
    DEFAULT_K,
    EXPORT,
    INC_BACKGROUND,
    INC_HEADLINE,
    Problem2D,
)


EDIT_ACTIONS = (
    INC_HEADLINE,
    DEC_HEADLINE,
    INC_BACKGROUND,
    DEC_BACKGROUND,
)


def _softmax_stable(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _validate_eps(eps):
    eps = float(eps)
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps must be in [0,1], got {eps!r}.")
    return eps


def _validate_gamma(gamma, *, allow_one):
    gamma = float(gamma)
    if allow_one:
        ok = 0.0 < gamma <= 1.0
        interval = "(0,1]"
    else:
        ok = 0.0 < gamma < 1.0
        interval = "(0,1)"
    if not ok:
        raise ValueError(f"gamma must be in {interval}, got {gamma!r}.")
    return gamma


def _validate_positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return value


def _validate_nonnegative_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.") from None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    if out < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
    return out


def _trembling_action_probs(available_actions, intended_actions, eps):
    eps = _validate_eps(eps)
    available = list(available_actions)
    intended = [a for a in available if a in set(intended_actions)]
    if not intended:
        raise ValueError("intended_actions must contain at least one available action.")

    non_intended = [a for a in available if a not in set(intended)]
    probs = {a: 0.0 for a in available}

    if non_intended:
        intended_mass = 1.0 - eps
        non_intended_mass = eps
    else:
        intended_mass = 1.0
        non_intended_mass = 0.0

    for action in intended:
        probs[action] = intended_mass / float(len(intended))
    for action in non_intended:
        probs[action] = non_intended_mass / float(len(non_intended))

    return probs


class _Base2DUserModel:
    def __init__(self, utility_fn, k=DEFAULT_K, rng=None):
        self.utility_fn = utility_fn
        self.problem = Problem2D(k=k)
        self.k = self.problem.k
        self.rng = np.random.default_rng() if rng is None else rng

        self.posters = np.asarray(self.problem.all_posters(), dtype=int)
        self.states = np.asarray(
            [self.problem.from_poster(poster) for poster in self.posters],
            dtype=int,
        )
        self.U = self._evaluate_utility_fn()

    def _evaluate_utility_fn(self):
        U = np.asarray(self.utility_fn(self.posters), dtype=float).reshape(-1)
        if U.shape != (self.problem.num_states,):
            raise ValueError(
                "utility_fn must return one utility value per poster; "
                f"expected {self.problem.num_states}, got shape {U.shape}."
            )
        if np.any(~np.isfinite(U)):
            raise ValueError("utility_fn returned non-finite utility values.")
        return U

    def _idx(self, poster):
        return self.problem.state_index(poster)

    def _poster_from_idx(self, idx):
        return self.problem.poster_from_index(idx)

    def _distance_from_state(self, h, b):
        return np.abs(self.states[:, 0] - int(h)) + np.abs(self.states[:, 1] - int(b))

    def _distance(self, poster_a, poster_b):
        h_a, b_a = self.problem.from_poster(poster_a)
        h_b, b_b = self.problem.from_poster(poster_b)
        return abs(h_a - h_b) + abs(b_a - b_b)

    def _shortest_path_actions(self, poster, target_poster):
        if int(poster) == int(target_poster):
            return []

        current_dist = self._distance(poster, target_poster)
        intended = []
        for action in self.problem.available_actions(poster):
            if action == EXPORT:
                continue
            next_poster, _done = self.problem.step(poster, action)
            if self._distance(next_poster, target_poster) == current_dist - 1:
                intended.append(action)
        return intended

    def _sample_action(self, probs):
        actions = list(probs.keys())
        p = np.asarray([probs[action] for action in actions], dtype=float)
        return str(self.rng.choice(actions, p=p))


class UserModel1GoalDirected2D(_Base2DUserModel):
    def __init__(self, utility_fn, k=DEFAULT_K, eps=0.0, rng=None):
        super().__init__(utility_fn=utility_fn, k=k, rng=rng)
        self.eps = _validate_eps(eps)
        self.target_idx = int(np.argmax(self.U))
        self.target_poster = self._poster_from_idx(self.target_idx)

    def intended_actions(self, poster):
        poster = int(poster)
        if poster == self.target_poster:
            return [EXPORT]
        return self._shortest_path_actions(poster, self.target_poster)

    def action_probs(self, poster):
        return _trembling_action_probs(
            self.problem.available_actions(poster),
            self.intended_actions(poster),
            self.eps,
        )

    def pick_action(self, poster):
        return self._sample_action(self.action_probs(poster))


class UserModel2Discounted2D(_Base2DUserModel):
    def __init__(self, utility_fn, gamma=0.95, k=DEFAULT_K, eps=0.0, rng=None):
        super().__init__(utility_fn=utility_fn, k=k, rng=rng)
        self.gamma = _validate_gamma(gamma, allow_one=True)
        self.eps = _validate_eps(eps)

    def _pick_target(self, poster):
        h, b = self.problem.from_poster(poster)
        distances = self._distance_from_state(h, b)
        scores = (self.gamma ** distances) * self.U
        return self._poster_from_idx(int(np.argmax(scores)))

    def intended_actions(self, poster):
        poster = int(poster)
        target = self._pick_target(poster)
        if poster == target:
            return [EXPORT]
        return self._shortest_path_actions(poster, target)

    def action_probs(self, poster):
        return _trembling_action_probs(
            self.problem.available_actions(poster),
            self.intended_actions(poster),
            self.eps,
        )

    def pick_action(self, poster):
        return self._sample_action(self.action_probs(poster))


class UserModel3Lookahead2D(_Base2DUserModel):
    def __init__(self, utility_fn, lookahead_k=3, k=DEFAULT_K, eps=0.0, rng=None):
        super().__init__(utility_fn=utility_fn, k=k, rng=rng)
        self.lookahead_k = _validate_nonnegative_int(lookahead_k, "lookahead_k")
        self.eps = _validate_eps(eps)

    def _pick_visible_target(self, poster):
        idx = self._idx(poster)
        h, b = self.problem.from_poster(poster)
        distances = self._distance_from_state(h, b)
        visible_idx = np.flatnonzero(distances <= self.lookahead_k)
        best_visible_utility = float(np.max(self.U[visible_idx]))

        if self.U[idx] >= best_visible_utility:
            return int(poster)

        local_best_pos = int(np.argmax(self.U[visible_idx]))
        return self._poster_from_idx(int(visible_idx[local_best_pos]))

    def intended_actions(self, poster):
        poster = int(poster)
        target = self._pick_visible_target(poster)
        if poster == target:
            return [EXPORT]
        return self._shortest_path_actions(poster, target)

    def action_probs(self, poster):
        return _trembling_action_probs(
            self.problem.available_actions(poster),
            self.intended_actions(poster),
            self.eps,
        )

    def pick_action(self, poster):
        return self._sample_action(self.action_probs(poster))


class UserModel4Boltzmann2D(_Base2DUserModel):
    def __init__(
        self,
        utility_fn,
        gamma=0.95,
        alpha=5.0,
        k=DEFAULT_K,
        rng=None,
        vi_tol=1e-12,
        vi_max_iter=200000,
    ):
        super().__init__(utility_fn=utility_fn, k=k, rng=rng)
        self.gamma = _validate_gamma(gamma, allow_one=False)
        self.alpha = _validate_positive(alpha, "alpha")
        self.vi_tol = _validate_positive(vi_tol, "vi_tol")
        self.vi_max_iter = _validate_nonnegative_int(vi_max_iter, "vi_max_iter")
        if self.vi_max_iter <= 0:
            raise ValueError(f"vi_max_iter must be positive, got {vi_max_iter!r}.")

        self.V = self._value_iteration()
        self.Q = self._compute_qstar()

    def _value_iteration(self):
        V = np.zeros(self.problem.num_states, dtype=float)

        for _ in range(self.vi_max_iter):
            V_old = V.copy()
            for idx, poster in enumerate(self.posters):
                best = self.U[idx]
                for action in self.problem.available_actions(int(poster)):
                    if action == EXPORT:
                        continue
                    next_poster, _done = self.problem.step(int(poster), action)
                    next_idx = self._idx(next_poster)
                    best = max(best, self.gamma * V_old[next_idx])
                V[idx] = best

            if np.max(np.abs(V - V_old)) < self.vi_tol:
                return V

        raise RuntimeError(
            f"Value iteration did not converge in {self.vi_max_iter} iterations."
        )

    def _compute_qstar(self):
        Q = {
            EXPORT: np.zeros(self.problem.num_states, dtype=float),
            INC_HEADLINE: np.full(self.problem.num_states, -np.inf, dtype=float),
            DEC_HEADLINE: np.full(self.problem.num_states, -np.inf, dtype=float),
            INC_BACKGROUND: np.full(self.problem.num_states, -np.inf, dtype=float),
            DEC_BACKGROUND: np.full(self.problem.num_states, -np.inf, dtype=float),
        }

        for idx, poster in enumerate(self.posters):
            poster = int(poster)
            Q[EXPORT][idx] = self.U[idx]
            for action in self.problem.available_actions(poster):
                if action == EXPORT:
                    continue
                next_poster, _done = self.problem.step(poster, action)
                Q[action][idx] = self.gamma * self.V[self._idx(next_poster)]

        return Q

    def action_probs(self, poster):
        idx = self._idx(poster)
        actions = self.problem.available_actions(poster)
        qvals = np.asarray([self.Q[action][idx] for action in actions], dtype=float)
        p = _softmax_stable(self.alpha * qvals)
        return {action: float(prob) for action, prob in zip(actions, p)}

    def pick_action(self, poster):
        return self._sample_action(self.action_probs(poster))
