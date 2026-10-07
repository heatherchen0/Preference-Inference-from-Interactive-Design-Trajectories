DEFAULT_K = 10

EXPORT = "export"
INC_HEADLINE = "inc-headline"
DEC_HEADLINE = "dec-headline"
INC_BACKGROUND = "inc-background"
DEC_BACKGROUND = "dec-background"


def _as_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None


class Problem2D:
    """
    Two-dimensional grayscale poster editor.

    Coordinates are one-based: (h, b) in {1, ..., k} x {1, ..., k}.
    The public poster id is also one-based:
        poster = (h - 1) * k + b
    """

    def __init__(self, k=DEFAULT_K, h0=1, b0=1):
        self.k = _as_int(k, "k")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {k!r}.")

        self.h0 = _as_int(h0, "h0")
        self.b0 = _as_int(b0, "b0")
        self._validate_coord(self.h0, "h0")
        self._validate_coord(self.b0, "b0")

        self.poster_min = 1
        self.poster_max = self.k * self.k
        self.num_states = self.poster_max

    def _validate_coord(self, value, name):
        v = _as_int(value, name)
        if v < 1 or v > self.k:
            raise ValueError(f"{name}={value!r} out of range [1,{self.k}].")

    def _validate_poster(self, poster):
        p = _as_int(poster, "poster")
        if p < self.poster_min or p > self.poster_max:
            raise ValueError(
                f"poster {poster!r} out of range "
                f"[{self.poster_min},{self.poster_max}]."
            )

    def sample_initial_poster(self):
        return self.to_poster(self.h0, self.b0)

    def to_poster(self, h, b):
        h = _as_int(h, "h")
        b = _as_int(b, "b")
        self._validate_coord(h, "h")
        self._validate_coord(b, "b")
        return (h - 1) * self.k + b

    def from_poster(self, poster):
        p = _as_int(poster, "poster")
        self._validate_poster(p)
        zero_based = p - 1
        h = zero_based // self.k + 1
        b = zero_based % self.k + 1
        return int(h), int(b)

    def state_index(self, poster):
        p = _as_int(poster, "poster")
        self._validate_poster(p)
        return p - 1

    def poster_from_index(self, index):
        i = _as_int(index, "state index")
        if i < 0 or i >= self.num_states:
            raise ValueError(f"state index {index!r} out of range [0,{self.num_states - 1}].")
        return i + 1

    def all_posters(self):
        return list(range(self.poster_min, self.poster_max + 1))

    def all_states(self):
        return [self.from_poster(poster) for poster in self.all_posters()]

    def available_actions(self, poster):
        h, b = self.from_poster(poster)
        actions = [EXPORT]

        if h < self.k:
            actions.append(INC_HEADLINE)
        if h > 1:
            actions.append(DEC_HEADLINE)
        if b < self.k:
            actions.append(INC_BACKGROUND)
        if b > 1:
            actions.append(DEC_BACKGROUND)

        return actions

    def step(self, poster, action):
        h, b = self.from_poster(poster)
        allowed = self.available_actions(poster)
        if action not in allowed:
            raise ValueError(
                f"Action {action!r} not allowed at poster={int(poster)} "
                f"(h={h}, b={b}). Allowed: {allowed}"
            )

        if action == EXPORT:
            return self.to_poster(h, b), True
        if action == INC_HEADLINE:
            return self.to_poster(h + 1, b), False
        if action == DEC_HEADLINE:
            return self.to_poster(h - 1, b), False
        if action == INC_BACKGROUND:
            return self.to_poster(h, b + 1), False
        if action == DEC_BACKGROUND:
            return self.to_poster(h, b - 1), False

        raise ValueError(f"Unknown action: {action!r}")
