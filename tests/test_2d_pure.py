from __future__ import annotations

import unittest

import numpy as np

from src.metrics import compute_contrast_profile_metrics, compute_recommendation_metrics
from src.preprocess import (
    prepare_inverse_data,
    prepare_pbo_start_end_data,
    prepare_pbo_transition_data,
)
from src.state_space import (
    ACTION_ORDER_2D,
    contrast_profile,
    neighbor_indices_and_mask_2d,
    normalized_gp_inputs,
)


class Pure2DInferenceTests(unittest.TestCase):
    def test_prepare_inverse_data_keeps_1d_default_action_mapping(self):
        rows = [
            {"poster": 1, "action": "dec"},
            {"poster": 2, "action": "export"},
            {"poster": 3, "action": "inc"},
        ]
        state_idx, action_idx, num_states = prepare_inverse_data(rows, poster_min=1, poster_max=3)
        self.assertEqual(num_states, 3)
        self.assertEqual(state_idx.tolist(), [0, 1, 2])
        self.assertEqual(action_idx.tolist(), [0, 1, 2])

    def test_prepare_inverse_data_maps_2d_actions_in_declared_order(self):
        rows = [
            {"poster": 1, "action": "export"},
            {"poster": 1, "action": "inc-headline"},
            {"poster": 11, "action": "dec-headline"},
            {"poster": 1, "action": "inc-background"},
            {"poster": 2, "action": "dec-background"},
        ]
        _state_idx, action_idx, num_states = prepare_inverse_data(
            rows,
            poster_min=1,
            poster_max=100,
            state_space="2d",
        )
        self.assertEqual(num_states, 100)
        self.assertEqual(tuple(ACTION_ORDER_2D), (
            "export",
            "inc-headline",
            "dec-headline",
            "inc-background",
            "dec-background",
        ))
        self.assertEqual(action_idx.tolist(), [0, 1, 2, 3, 4])

    def test_2d_neighbors_mask_boundaries(self):
        neighbors, valid = neighbor_indices_and_mask_2d(grid_k=3)
        self.assertEqual(neighbors.shape, (9, 5))
        self.assertEqual(valid.shape, (9, 5))
        self.assertEqual(valid[0].tolist(), [True, True, False, True, False])
        self.assertEqual(neighbors[0].tolist(), [0, 3, 0, 1, 0])
        self.assertEqual(valid[4].tolist(), [True, True, True, True, True])

    def test_2d_gp_inputs_use_coordinates_not_flattened_line(self):
        X = normalized_gp_inputs(state_space="2d", num_states=100, grid_k=10)
        self.assertEqual(X.shape, (100, 2))
        np.testing.assert_allclose(X[0], [0.0, 0.0])
        np.testing.assert_allclose(X[9], [0.0, 1.0])
        np.testing.assert_allclose(X[10], [1.0 / 9.0, 0.0])
        np.testing.assert_allclose(X[99], [1.0, 1.0])

    def test_pbo_comparisons_include_2d_coordinates(self):
        rows = [
            {"traj_id": 0, "t": 0, "poster": 1, "action": "inc-background"},
            {"traj_id": 0, "t": 1, "poster": 2, "action": "export"},
            {"traj_id": 1, "t": 0, "poster": 11, "action": "export"},
        ]
        start_idx, end_idx, _S, stats, cmp_rows = prepare_pbo_start_end_data(
            rows,
            poster_min=1,
            poster_max=100,
            state_space="2d",
            grid_k=10,
        )
        self.assertEqual(start_idx.tolist(), [0])
        self.assertEqual(end_idx.tolist(), [1])
        self.assertEqual(stats["n_ties_dropped"], 1)
        self.assertEqual(cmp_rows[0]["start_headline_gray"], 1)
        self.assertEqual(cmp_rows[0]["end_background_gray"], 2)

        tr_start, tr_end, _S, tr_stats, tr_cmp = prepare_pbo_transition_data(
            rows,
            poster_min=1,
            poster_max=100,
            state_space="2d",
            grid_k=10,
        )
        self.assertEqual(tr_start.tolist(), [0])
        self.assertEqual(tr_end.tolist(), [1])
        self.assertEqual(tr_stats["n_transition_candidates"], 1)
        self.assertEqual(tr_cmp[0]["end_background_gray"], 2)

    def test_contrast_metrics_and_recommendation_fields(self):
        U_true = np.arange(1, 10, dtype=float)
        U_pred = U_true.copy()
        metrics = compute_contrast_profile_metrics(U_true=U_true, U_pred=U_pred, grid_k=3)
        self.assertEqual(metrics["contrast_profile_rmse"], 0.0)
        np.testing.assert_allclose(contrast_profile(U_true, grid_k=3), [5.0, 5.0, 5.0])

        rec = compute_recommendation_metrics(
            posters=np.arange(1, 10),
            U_true=U_true,
            recommendation_score=U_pred,
            state_space="2d",
            grid_k=3,
            preferred_contrast=2,
        )
        self.assertEqual(rec["g_hat"], 9)
        self.assertEqual(rec["h_hat"], 3)
        self.assertEqual(rec["b_hat"], 3)
        self.assertEqual(rec["c_hat"], 0)
        self.assertEqual(rec["c_star"], 2)
        self.assertEqual(rec["contrast_error"], 2.0)


if __name__ == "__main__":
    unittest.main()
