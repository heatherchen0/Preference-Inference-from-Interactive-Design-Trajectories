G_MIN = 1
G_MAX = 100

INC = "inc"
DEC = "dec"
EXPORT = "export"

class Problem1D:
    def __init__(self, g_min=G_MIN, g_max=G_MAX, g0=1):
        self.g_min = int(g_min)
        self.g_max = int(g_max)
        self.g0 = int(g0)

    def sample_initial_poster(self):
        return self.g0

    def available_actions(self, poster):
        g = int(poster)
        actions = [EXPORT]
        if g < self.g_max:
            actions.append(INC)
        if g > self.g_min:
            actions.append(DEC)
        return actions

    def step(self, poster, action):
        g = int(poster)
        allowed = self.available_actions(g)
        if action not in allowed:
            raise ValueError(f"Action '{action}' not allowed at g={g}. Allowed: {allowed}")

        if action == INC:
            return g + 1, False
        if action == DEC:
            return g - 1, False
        if action == EXPORT:
            return g, True

        raise ValueError(f"Unknown action: {action}")