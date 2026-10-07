from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from src.config import ExperimentConfig, validate_experiment_config
from src.metrics import summarize_posterior_var


class _Var:
    def __init__(self, values):
        self.values = values


class _FakeIdata:
    def __init__(self, **posterior):
        self.posterior = {k: _Var(v) for k, v in posterior.items()}


class InferAlphaTests(unittest.TestCase):
    def test_config_accepts_infer_alpha_in_1d(self):
        cfg = ExperimentConfig(
            model_name="model4_endpoint_discounted_boltzmann_infer_alpha",
            state_space="1d",
            alpha_prior_median=5.0,
            alpha_prior_log_sd=0.75,
        )
        validate_experiment_config(cfg)

    def test_config_rejects_infer_alpha_in_2d(self):
        cfg = ExperimentConfig(
            model_name="model4_endpoint_discounted_boltzmann_infer_alpha",
            state_space="2d",
            poster_min=1,
            poster_max=100,
            grid_k=10,
            utility_kind="smooth_contrast_interaction",
        )
        with self.assertRaises(ValueError):
            validate_experiment_config(cfg)

    def test_scalar_alpha_summary(self):
        idata = _FakeIdata(alpha=np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]))
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "arviz":
                raise ImportError("force quantile fallback")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            summary = summarize_posterior_var(
                idata,
                var_name="alpha",
                hdi_prob=0.5,
                normalize_draws=False,
            )
        self.assertAlmostEqual(float(summary["mean"]), 7.0)
        self.assertGreater(float(summary["sd"]), 0.0)
        self.assertIn("low", summary)
        self.assertIn("high", summary)

    def test_variant_selection_for_infer_alpha(self):
        run_inference = _import_or_skip("run_inference")
        selected, variants = run_inference._select_variants(
            variant_set="infer_alpha",
            dataset_suite="main",
            alpha_values=None,
        )
        self.assertEqual(selected, "infer_alpha")
        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[0]["variant_label"], "m4-boltz-gp_rbf-alpha5")
        self.assertEqual(
            variants[1]["model_name"],
            "model4_endpoint_discounted_boltzmann_infer_alpha",
        )
        self.assertEqual(
            run_inference._variant_label_from_spec(variants[1]),
            "m4-infer_alpha-boltz-gp_rbf",
        )

    def test_infer_alpha_triton_task_generation(self):
        task_mod = _import_or_skip("triton_make_infer_alpha_tasks")
        main_tasks = task_mod.make_main_tasks()
        robustness_tasks = task_mod.make_robustness_tasks()
        self.assertEqual(len(main_tasks), 10)
        self.assertEqual(len(robustness_tasks), 10)
        self.assertEqual(main_tasks[0], ("smooth", 0, 1))
        self.assertEqual(main_tasks[-1], ("jagged", 4, 1))
        self.assertEqual(robustness_tasks[0], (1, 0, 1))
        self.assertEqual(robustness_tasks[-1], (2, 4, 1))

    def test_infer_alpha_triton_runners_expose_one_variant(self):
        main_runner = _import_or_skip("triton_run_infer_alpha_1d_task")
        robustness_runner = _import_or_skip("triton_run_infer_alpha_robustness_task")
        for module in (main_runner, robustness_runner):
            self.assertEqual(len(module.VARIANTS), 1)
            self.assertEqual(
                module.VARIANTS[0]["model_name"],
                "model4_endpoint_discounted_boltzmann_infer_alpha",
            )
            self.assertEqual(
                module.VARIANTS[0]["variant_label"],
                "m4-infer_alpha-boltz-gp_rbf",
            )

    def test_builder_exposes_expected_variables(self):
        _import_or_skip("pymc")
        model_mod = _import_or_skip("src.model")
        model = model_mod.build_model4_endpoint_discounted_boltzmann_infer_alpha(
            state_idx=np.array([0, 1, 2]),
            action_idx=np.array([2, 2, 1]),
            num_states=4,
            gamma_fixed=0.95,
            alpha_prior_median=5.0,
            alpha_prior_log_sd=0.75,
            prior="iid",
        )
        for name in ("U", "alpha_raw", "log_alpha", "alpha", "V", "Q_all", "action_probs", "obs"):
            self.assertIn(name, model.named_vars)


def _import_or_skip(module_name: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return __import__(module_name, fromlist=["*"])
    except Exception as exc:
        raise unittest.SkipTest(f"{module_name} unavailable in this environment: {exc}") from exc


if __name__ == "__main__":
    unittest.main()
