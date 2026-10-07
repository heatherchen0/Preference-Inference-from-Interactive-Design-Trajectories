import numpy as np
from poster_env import INC, DEC, EXPORT

def _softmax_stable(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

class UserModel1:
    def __init__(self, utility_fn, g_min=1, g_max=100):
        self.utility_fn = utility_fn
        self.g_min = int(g_min)
        self.g_max = int(g_max)
        self.g_star = self._compute_global_optimum()

    def _compute_global_optimum(self):
        G = np.arange(self.g_min, self.g_max + 1)
        U = self.utility_fn(G)
        return int(G[np.argmax(U)])

    def pick_action(self, poster):
        g = int(poster)
        if g == self.g_star:
            return EXPORT
        elif g < self.g_star:
            return INC
        else:
            return DEC

class UserModel2Discounted:
    def __init__(self, utility_fn, gamma, g_min=1, g_max=100):
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        self.utility_fn = utility_fn
        self.gamma = float(gamma)
        self.g_min = int(g_min)
        self.g_max = int(g_max)

        self.G = np.arange(self.g_min, self.g_max + 1)
        self.U = self.utility_fn(self.G)

    def _pick_target(self, poster):
        g_t = int(poster)
        distances = np.abs(self.G - g_t)
        scores = (self.gamma ** distances) * self.U
        best_index = int(np.argmax(scores))
        return int(self.G[best_index])

    def pick_action(self, poster):
        g_t = int(poster)
        target = self._pick_target(g_t)

        if g_t == target:
            return EXPORT
        elif g_t < target:
            return INC
        else:
            return DEC

class UserModel3Lookahead:
    def __init__(self, utility_fn, k, g_min=1, g_max=100, export_on_tie=True):
        self.utility_fn = utility_fn
        self.k = int(k)
        if self.k < 0:
            raise ValueError("k must be >= 0")

        self.g_min = int(g_min)
        self.g_max = int(g_max)
        if self.g_min > self.g_max:
            raise ValueError("g_min must be <= g_max")

        self.export_on_tie = bool(export_on_tie)

        self.G = list(range(self.g_min, self.g_max + 1))
        U = self.utility_fn(self.G)
        if len(U) != len(self.G):
            raise ValueError("utility_fn must return a utility for each g in the grid")
        self.U_by_g = {g: float(u) for g, u in zip(self.G, U)}

    def _window(self, g):
        g = int(g)
        lo = max(self.g_min, g - self.k)
        hi = min(self.g_max, g + self.k)
        return lo, hi

    def _argmax_in_window(self, g):
        lo, hi = self._window(g)
        best_val = None
        best = []
        for h in range(lo, hi + 1):
            u = self.U_by_g[h]
            if best_val is None or u > best_val:
                best_val = u
                best = [h]
            elif u == best_val:
                best.append(h)
        return best_val, best

    def pick_action(self, poster):
        g = int(poster)

        if self.k == 0:
            return EXPORT

        _, best_set = self._argmax_in_window(g)

        if self.export_on_tie and (g in best_set):
            return EXPORT

        target = min(best_set, key=lambda h: (abs(h - g), h))

        if target == g:
            return EXPORT
        if target > g:
            return INC
        return DEC

class UserModel4_EndpointDiscountedBoltzmann:
    """
    Endpoint-only reward (only export gives U(g)), discounting provides urgency,
    Boltzmann-rational actions: P(a|g) ∝ exp(alpha * Q*(g,a)).
    """
    def __init__(
        self,
        utility_fn,
        g_min=1,
        g_max=100,
        alpha=5.0,
        gamma=0.995,
        rng=None,
        vi_tol=1e-12,
        vi_max_iter=200000,
    ):
        self.utility_fn = utility_fn
        self.g_min = int(g_min)
        self.g_max = int(g_max)
        self.n = self.g_max - self.g_min + 1

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma must be in (0,1), got {self.gamma}")

        self.rng = np.random.default_rng() if rng is None else rng
        self.vi_tol = float(vi_tol)
        self.vi_max_iter = int(vi_max_iter)

        gs = np.arange(self.g_min, self.g_max + 1, dtype=float)
        U = np.asarray(self.utility_fn(gs), dtype=float).reshape((self.n,))
        self.U = U

        self.V = self._value_iteration()
        self.Q = self._compute_qstar()

    def _idx(self, g):
        return int(g) - self.g_min

    def _value_iteration(self):
        gamma = self.gamma
        V = np.zeros(self.n, dtype=float)

        for _ in range(self.vi_max_iter):
            V_old = V.copy()
            for i in range(self.n):
                best = self.U[i]  # export
                if i < self.n - 1:
                    best = max(best, gamma * V_old[i + 1])
                if i > 0:
                    best = max(best, gamma * V_old[i - 1])
                V[i] = best

            if np.max(np.abs(V - V_old)) < self.vi_tol:
                return V

        raise RuntimeError(f"Value iteration did not converge in {self.vi_max_iter} iterations")

    def _compute_qstar(self):
        gamma = self.gamma
        Q = {
            EXPORT: np.zeros(self.n, dtype=float),
            INC:    np.full(self.n, -np.inf, dtype=float),
            DEC:    np.full(self.n, -np.inf, dtype=float),
        }
        for i in range(self.n):
            Q[EXPORT][i] = self.U[i]
            if i < self.n - 1:
                Q[INC][i] = gamma * self.V[i + 1]
            if i > 0:
                Q[DEC][i] = gamma * self.V[i - 1]
        return Q

    def pick_action(self, poster):
        g = int(poster)
        i = self._idx(g)

        actions = [EXPORT]
        if g < self.g_max:
            actions.append(INC)
        if g > self.g_min:
            actions.append(DEC)

        qvals = np.array([self.Q[a][i] for a in actions], dtype=float)
        probs = _softmax_stable(self.alpha * qvals)
        return self.rng.choice(actions, p=probs)