if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from poster2d import (
    coord_to_1d_gray,
    DEC_BACKGROUND,
    DEC_HEADLINE,
    EXPORT,
    INC_BACKGROUND,
    INC_HEADLINE,
    Problem2D,
    smooth_contrast_interaction_grid,
    smooth_contrast_interaction_utility,
    UserModel1GoalDirected2D,
    UserModel2Discounted2D,
    UserModel3Lookahead2D,
    UserModel4Boltzmann2D,
)
from poster2d.utility import PREFERRED_CONTRAST
from poster2d.simulate import (
    materialize_step_records,
    run_systematic_sweep_2d,
    simulate_trajectory_from_start,
)


def _assert_raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def _run_plotting_smoke_check(U_grid):
    import contextlib
    import io

    try:
        import_stderr = io.StringIO()
        with contextlib.redirect_stderr(import_stderr):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            from poster2d.plotting import (
                plot_smooth_contrast_interaction_utility,
                plot_utility_heatmap,
            )
    except (AttributeError, ImportError) as exc:
        print(f"plotting smoke check skipped: {exc}")
        return

    fig, ax = plot_utility_heatmap(U_grid, show=False)
    assert fig is ax.figure
    plt.close(fig)

    fig, ax = plot_smooth_contrast_interaction_utility(show=False)
    assert fig is ax.figure
    plt.close(fig)


def _assert_probs_normalized(probs):
    values = np.asarray(list(probs.values()), dtype=float)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.isclose(float(np.sum(values)), 1.0)


def _cardinal_neighbor_values(U_grid, h, b):
    values = []
    for dh, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        hh = int(h) + dh
        bb = int(b) + db
        if 1 <= hh <= U_grid.shape[0] and 1 <= bb <= U_grid.shape[1]:
            values.append(float(U_grid[hh - 1, bb - 1]))
    return values


def _is_strict_cardinal_local_max(U_grid, h, b):
    center = float(U_grid[int(h) - 1, int(b) - 1])
    return all(center > neighbor for neighbor in _cardinal_neighbor_values(U_grid, h, b))


def _point_utility_fn(problem, values_by_state):
    def utility_fn(posters):
        poster_arr = np.asarray(posters)
        flat = poster_arr.reshape(-1)
        values = []
        for poster in flat:
            h, b = problem.from_poster(int(poster))
            values.append(float(values_by_state.get((h, b), 0.0)))
        return np.asarray(values, dtype=float).reshape(poster_arr.shape)

    return utility_fn


def _constant_utility(value):
    def utility_fn(posters):
        return np.full(np.asarray(posters).shape, float(value), dtype=float)

    return utility_fn


def _run_user_model_smoke_checks():
    problem = Problem2D(k=5)

    target_3_3 = problem.to_poster(3, 3)
    peak_3_3 = _point_utility_fn(problem, {(3, 3): 1.0})

    model1 = UserModel1GoalDirected2D(
        peak_3_3,
        k=5,
        rng=np.random.default_rng(1),
    )
    assert model1.target_poster == target_3_3
    probs = model1.action_probs(target_3_3)
    _assert_probs_normalized(probs)
    assert probs[EXPORT] == 1.0

    probs = model1.action_probs(problem.to_poster(1, 1))
    _assert_probs_normalized(probs)
    assert probs[INC_HEADLINE] == 0.5
    assert probs[INC_BACKGROUND] == 0.5
    assert probs[EXPORT] == 0.0

    probs = model1.action_probs(problem.to_poster(2, 3))
    _assert_probs_normalized(probs)
    assert probs[INC_HEADLINE] == 1.0

    eps_model1 = UserModel1GoalDirected2D(peak_3_3, k=5, eps=0.2)
    probs = eps_model1.action_probs(problem.to_poster(1, 1))
    _assert_probs_normalized(probs)
    assert np.isclose(probs[EXPORT], 0.2)
    assert np.isclose(probs[INC_HEADLINE], 0.4)
    assert np.isclose(probs[INC_BACKGROUND], 0.4)

    start = problem.to_poster(1, 1)
    far = problem.to_poster(5, 5)
    current_vs_far = _point_utility_fn(problem, {(1, 1): 0.9, (5, 5): 1.0})

    low_gamma_model2 = UserModel2Discounted2D(current_vs_far, k=5, gamma=0.5)
    probs = low_gamma_model2.action_probs(start)
    _assert_probs_normalized(probs)
    assert probs[EXPORT] == 1.0
    assert probs[INC_HEADLINE] == 0.0
    assert probs[INC_BACKGROUND] == 0.0

    high_gamma_model2 = UserModel2Discounted2D(current_vs_far, k=5, gamma=0.99)
    probs = high_gamma_model2.action_probs(start)
    _assert_probs_normalized(probs)
    assert probs[INC_HEADLINE] == 0.5
    assert probs[INC_BACKGROUND] == 0.5
    assert probs[EXPORT] == 0.0

    diagonal_local_peak = _point_utility_fn(problem, {(2, 2): 1.0})
    model3 = UserModel3Lookahead2D(diagonal_local_peak, k=5, lookahead_k=2)
    probs = model3.action_probs(start)
    _assert_probs_normalized(probs)
    assert probs[INC_HEADLINE] == 0.5
    assert probs[INC_BACKGROUND] == 0.5

    local_best_current = _point_utility_fn(problem, {(1, 1): 1.0, (2, 1): 0.5})
    model3_export = UserModel3Lookahead2D(local_best_current, k=5, lookahead_k=1)
    probs = model3_export.action_probs(start)
    _assert_probs_normalized(probs)
    assert probs[EXPORT] == 1.0
    assert probs[INC_HEADLINE] == 0.0
    assert probs[INC_BACKGROUND] == 0.0

    local_neighbor_peak = _point_utility_fn(problem, {(2, 1): 1.0})
    model3_move = UserModel3Lookahead2D(local_neighbor_peak, k=5, lookahead_k=1)
    probs = model3_move.action_probs(start)
    _assert_probs_normalized(probs)
    assert probs[INC_HEADLINE] == 1.0

    model4 = UserModel4Boltzmann2D(
        peak_3_3,
        k=5,
        gamma=0.95,
        alpha=5.0,
        rng=np.random.default_rng(2),
    )
    assert model4.V.shape == (25,)
    assert np.all(np.isfinite(model4.V))
    assert set(model4.Q) == {
        EXPORT,
        INC_HEADLINE,
        DEC_HEADLINE,
        INC_BACKGROUND,
        DEC_BACKGROUND,
    }
    assert model4.Q[EXPORT][problem.state_index(target_3_3)] == 1.0
    assert np.isneginf(model4.Q[DEC_HEADLINE][problem.state_index(start)])
    assert np.isneginf(model4.Q[DEC_BACKGROUND][problem.state_index(start)])

    for poster in problem.all_posters():
        probs = model4.action_probs(poster)
        _assert_probs_normalized(probs)
        assert list(probs.keys()) == problem.available_actions(poster)

    for _ in range(20):
        action = model4.pick_action(start)
        assert action in problem.available_actions(start)


def _assert_jsonl_ready_rows(problem, rows):
    required = {
        "traj_id",
        "t",
        "poster",
        "headline_gray",
        "background_gray",
        "action",
        "start_poster",
        "start_headline_gray",
        "start_background_gray",
        "rep",
        "model",
        "utility",
    }
    assert rows
    for row in rows:
        assert required.issubset(row.keys())
        h, b = problem.from_poster(row["poster"])
        assert row["headline_gray"] == h
        assert row["background_gray"] == b


def _assert_trajectory_rows_are_ordered(rows):
    by_tid = {}
    for row in rows:
        by_tid.setdefault(int(row["traj_id"]), []).append(row)

    for traj_rows in by_tid.values():
        ts = [int(row["t"]) for row in traj_rows]
        assert ts == list(range(len(ts)))
        assert traj_rows[-1]["action"] == EXPORT


def _run_simulation_smoke_checks():
    problem = Problem2D(k=10)
    starts = problem.all_posters()
    utility_fn = _point_utility_fn(problem, {(10, 1): 1.0})

    model1 = UserModel1GoalDirected2D(
        utility_fn,
        k=10,
        rng=np.random.default_rng(11),
    )
    one_traj = simulate_trajectory_from_start(
        problem,
        model1,
        utility_fn,
        start_poster=problem.to_poster(1, 1),
        max_steps=100,
    )
    assert one_traj[-1]["action"] == EXPORT
    for row in one_traj:
        h, b = problem.from_poster(row["poster"])
        assert row["headline_gray"] == h
        assert row["background_gray"] == b

    deterministic_models = [
        UserModel1GoalDirected2D(utility_fn, k=10, rng=np.random.default_rng(21)),
        UserModel2Discounted2D(utility_fn, k=10, gamma=0.99, rng=np.random.default_rng(22)),
        UserModel3Lookahead2D(utility_fn, k=10, lookahead_k=3, rng=np.random.default_rng(23)),
    ]
    for i, model in enumerate(deterministic_models, start=1):
        sweep = run_systematic_sweep_2d(
            problem,
            model,
            utility_fn,
            starts=starts,
            n_reps_per_start=1,
            max_steps=100,
            shuffle_trajectories=True,
            rng=np.random.default_rng(100 + i),
        )
        assert len(sweep["trajectories"]) == 100
        rows, per_traj_rows = materialize_step_records(
            sweep["trajectories"],
            model_label=f"smoke_model{i}",
            config_fields={"smoke": True},
        )
        assert len(per_traj_rows) == 100
        _assert_jsonl_ready_rows(problem, rows)
        _assert_trajectory_rows_are_ordered(rows)

    model4 = UserModel4Boltzmann2D(
        _constant_utility(1.0),
        k=10,
        gamma=0.95,
        alpha=50.0,
        rng=np.random.default_rng(31),
    )
    sweep = run_systematic_sweep_2d(
        problem,
        model4,
        _constant_utility(1.0),
        starts=starts,
        n_reps_per_start=15,
        max_steps=500,
        shuffle_trajectories=True,
        rng=np.random.default_rng(131),
    )
    assert len(sweep["trajectories"]) == 100 * 15
    rows, per_traj_rows = materialize_step_records(
        sweep["trajectories"],
        model_label="smoke_model4",
        config_fields={"gamma": 0.95, "alpha": 50.0},
    )
    assert len(per_traj_rows) == 100 * 15
    _assert_jsonl_ready_rows(problem, rows)
    _assert_trajectory_rows_are_ordered(rows)


def run_smoke_checks():
    problem = Problem2D()

    assert problem.k == 10
    assert problem.num_states == 100
    assert problem.sample_initial_poster() == 1

    assert problem.to_poster(1, 1) == 1
    assert problem.to_poster(1, 10) == 10
    assert problem.to_poster(2, 1) == 11
    assert problem.to_poster(10, 10) == 100

    for poster in problem.all_posters():
        h, b = problem.from_poster(poster)
        assert problem.to_poster(h, b) == poster
        assert problem.poster_from_index(problem.state_index(poster)) == poster

    assert problem.all_states()[0] == (1, 1)
    assert problem.all_states()[-1] == (10, 10)

    assert problem.available_actions(1) == [EXPORT, INC_HEADLINE, INC_BACKGROUND]
    assert problem.available_actions(10) == [
        EXPORT,
        INC_HEADLINE,
        DEC_BACKGROUND,
    ]
    assert problem.available_actions(91) == [
        EXPORT,
        DEC_HEADLINE,
        INC_BACKGROUND,
    ]
    assert problem.available_actions(100) == [
        EXPORT,
        DEC_HEADLINE,
        DEC_BACKGROUND,
    ]
    assert problem.available_actions(problem.to_poster(5, 5)) == [
        EXPORT,
        INC_HEADLINE,
        DEC_HEADLINE,
        INC_BACKGROUND,
        DEC_BACKGROUND,
    ]

    middle = problem.to_poster(5, 5)
    assert problem.step(middle, INC_HEADLINE) == (problem.to_poster(6, 5), False)
    assert problem.step(middle, DEC_HEADLINE) == (problem.to_poster(4, 5), False)
    assert problem.step(middle, INC_BACKGROUND) == (problem.to_poster(5, 6), False)
    assert problem.step(middle, DEC_BACKGROUND) == (problem.to_poster(5, 4), False)
    assert problem.step(middle, EXPORT) == (middle, True)

    _assert_raises_value_error(lambda: problem.to_poster(0, 1))
    _assert_raises_value_error(lambda: problem.to_poster(1.5, 1))
    _assert_raises_value_error(lambda: problem.to_poster(1, 11))
    _assert_raises_value_error(lambda: problem.from_poster(0))
    _assert_raises_value_error(lambda: problem.from_poster(1.5))
    _assert_raises_value_error(lambda: problem.from_poster(101))
    _assert_raises_value_error(lambda: problem.poster_from_index(-1))
    _assert_raises_value_error(lambda: problem.poster_from_index(1.5))
    _assert_raises_value_error(lambda: problem.poster_from_index(100))
    _assert_raises_value_error(lambda: Problem2D(k=0))
    _assert_raises_value_error(lambda: problem.step(1, DEC_HEADLINE))
    _assert_raises_value_error(lambda: problem.step(1, DEC_BACKGROUND))
    _assert_raises_value_error(lambda: problem.step(1, "unknown"))

    assert coord_to_1d_gray(1, k=10) == 1.0
    assert coord_to_1d_gray(10, k=10) == 100.0
    assert np.allclose(coord_to_1d_gray([1, 10], k=10), [1.0, 100.0])

    U_grid = smooth_contrast_interaction_grid()
    assert U_grid.shape == (10, 10)
    assert np.all(np.isfinite(U_grid))
    assert np.min(U_grid) >= -1e-12
    assert np.max(U_grid) <= 1.0 + 1e-12
    assert np.isclose(np.min(U_grid), 0.0)
    assert np.isclose(np.max(U_grid), 1.0)

    posters = np.arange(1, 101)
    values = smooth_contrast_interaction_utility(posters)
    assert values.shape == (100,)
    assert np.allclose(values, U_grid.reshape(-1))
    assert smooth_contrast_interaction_utility(1) == float(U_grid[0, 0])

    best_h_idx, best_b_idx = np.unravel_index(int(np.argmax(U_grid)), U_grid.shape)
    best_h = best_h_idx + 1
    best_b = best_b_idx + 1
    assert PREFERRED_CONTRAST == 5
    assert (best_h, best_b) == (8, 3)
    assert np.isclose(U_grid[7, 2], 1.0)

    local_peak = float(U_grid[3, 8])
    bump_peak = float(U_grid[3, 2])
    assert np.isclose(local_peak, 0.78, atol=0.03)
    assert 0.39 <= bump_peak <= 0.45
    assert bump_peak < local_peak

    global_neighbors = _cardinal_neighbor_values(U_grid, 8, 3)
    local_neighbors = _cardinal_neighbor_values(U_grid, 4, 9)
    assert np.mean(global_neighbors) > np.mean(local_neighbors)
    assert 0.83 <= np.mean(global_neighbors) <= 0.88
    assert 0.57 <= np.mean(local_neighbors) <= 0.63
    assert sum(v >= 0.6 for v in global_neighbors) > sum(
        v >= 0.6 for v in local_neighbors
    )

    assert _is_strict_cardinal_local_max(U_grid, 4, 9)
    assert _is_strict_cardinal_local_max(U_grid, 4, 3)
    bump_neighbors = _cardinal_neighbor_values(U_grid, 4, 3)
    assert sum(0.20 <= v < bump_peak for v in bump_neighbors) == 3

    _run_user_model_smoke_checks()
    _run_simulation_smoke_checks()

    _run_plotting_smoke_check(U_grid)


if __name__ == "__main__":
    run_smoke_checks()
    print("poster2d smoke checks passed")
