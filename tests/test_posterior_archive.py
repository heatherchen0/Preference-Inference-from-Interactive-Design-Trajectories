from __future__ import annotations

import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import numpy as np

from src.posterior_archive import (
    load_alpha_draw_archive,
    load_u_draw_archive,
    read_archive_manifest,
    save_alpha_draw_archive,
    save_u_draw_archive,
    summarize_idata_diagnostics,
    write_archive_manifest,
)


class _Var:
    def __init__(self, values):
        self.values = values


class _FakeIdata:
    def __init__(self, U=None, **posterior):
        if U is not None:
            posterior["U"] = U
        self.posterior = {k: _Var(v) for k, v in posterior.items()}


class PosteriorArchiveTests(unittest.TestCase):
    def test_u_draw_archive_round_trip_includes_checksums(self):
        U = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) / 10.0
        idata = _FakeIdata(U)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            info = save_u_draw_archive(
                idata=idata,
                out_path=root / "main_U_draws.npz",
                posters=[1, 2, 3, 4],
                U_true=[0.1, 0.2, 0.3, 0.4],
                metadata={"scope": "main"},
                dtype="float32",
            )

            loaded = load_u_draw_archive(root / "main_U_draws.npz")
            self.assertEqual(loaded["U"].shape, (2, 3, 4))
            self.assertEqual(str(loaded["U"].dtype), "float32")
            self.assertEqual(loaded["posters"].tolist(), [1, 2, 3, 4])
            self.assertEqual(loaded["metadata"]["scope"], "main")

            for key in ("U_sha256", "posters_sha256", "U_true_sha256", "archive_file_sha256"):
                self.assertIn(key, info)
                self.assertIsInstance(info[key], str)
                self.assertEqual(len(info[key]), 64)

            for key in ("U_sha256", "posters_sha256", "U_true_sha256", "archive_file_sha256"):
                self.assertIn(key, loaded["metadata"] if key != "archive_file_sha256" else loaded)

            manifest_path = write_archive_manifest(
                archive_dir=root,
                entries=[{"scope": "main", "kind": "u_draws", **info}],
                metadata={"ok": True},
            )
            manifest = read_archive_manifest(manifest_path)
            entry = manifest["entries"][0]
            self.assertEqual(entry["shape"], [2, 3, 4])
            self.assertEqual(entry["U_sha256"], info["U_sha256"])
            self.assertEqual(entry["archive_file_sha256"], info["archive_file_sha256"])

    def test_alpha_draw_archive_round_trip_includes_log_alpha(self):
        alpha = np.array([[4.0, 5.0, 6.0], [5.5, 6.5, 7.5]], dtype=float)
        log_alpha = np.log(alpha)
        idata = _FakeIdata(alpha=alpha, log_alpha=log_alpha)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            info = save_alpha_draw_archive(
                idata=idata,
                out_path=root / "alpha_draws" / "main_alpha_draws.npz",
                metadata={"scope": "main"},
                dtype="float32",
            )

            loaded = load_alpha_draw_archive(root / "alpha_draws" / "main_alpha_draws.npz")
            self.assertEqual(loaded["alpha"].shape, (2, 3))
            self.assertEqual(str(loaded["alpha"].dtype), "float32")
            self.assertEqual(loaded["metadata"]["scope"], "main")
            np.testing.assert_allclose(loaded["alpha"], alpha.astype("float32"))
            np.testing.assert_allclose(loaded["log_alpha"], log_alpha.astype("float32"))

            for key in ("alpha_sha256", "log_alpha_sha256", "archive_file_sha256"):
                self.assertIn(key, info)
                self.assertIsInstance(info[key], str)
                self.assertEqual(len(info[key]), 64)

    def test_metric_recomputation_from_archive_draws_matches_direct_draws(self):
        pp = _import_or_skip("src.postprocess")
        U = np.array(
            [
                [[0.0, 1.0, 2.0, 3.0], [0.2, 1.1, 2.1, 2.8]],
                [[0.1, 0.9, 1.9, 3.1], [0.3, 1.2, 2.2, 3.2]],
            ],
            dtype=np.float32,
        )
        posters = np.array([1, 2, 3, 4])
        U_true = np.array([0.0, 1.0, 2.0, 3.0])

        bundle = pp._summaries_and_metrics_from_draws(
            draws_chain=U,
            posters=posters,
            U_true=U_true,
            hdi_prob=0.5,
        )

        flat = U.reshape((-1, U.shape[-1])).astype(float)
        direct_mean = np.mean(flat, axis=0)
        direct_mse = float(np.mean((direct_mean - U_true) ** 2))

        np.testing.assert_allclose(bundle["raw"]["mean"], direct_mean)
        self.assertAlmostEqual(bundle["metrics_raw"]["mse"], direct_mse)
        self.assertEqual(bundle["ranking_posterior"]["n_draws"], flat.shape[0])

    def test_diagnostics_extraction_minimal_inferencedata(self):
        az = _import_or_skip("arviz")
        xr = _import_or_skip("xarray")

        posterior = xr.Dataset(
            {"U": (("chain", "draw", "state"), np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3))}
        )
        sample_stats = xr.Dataset(
            {
                "diverging": (("chain", "draw"), np.array([[False, True, False, False], [False, False, False, False]])),
                "acceptance_rate": (("chain", "draw"), np.full((2, 4), 0.91)),
                "tree_depth": (("chain", "draw"), np.array([[1, 2, 2, 3], [1, 3, 2, 2]])),
            }
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        diag, per_state = summarize_idata_diagnostics(idata, posters=[1, 2, 3])
        self.assertEqual(diag["divergence_count"], 1)
        self.assertEqual(diag["n_chains"], 2)
        self.assertEqual(diag["n_draws"], 4)
        self.assertEqual(len(per_state), 3)

    def test_postprocess_variant_does_not_sample(self):
        pp = _import_or_skip("src.postprocess")
        sampling = _import_or_skip("src.sampling")

        old_sample_model = sampling.sample_model
        sampling.sample_model = _raise_if_sampled
        try:
            with tempfile.TemporaryDirectory() as td:
                variant_dir = Path(td) / "v=fake"
                archive_dir = variant_dir / "posterior_archive"
                variant_dir.mkdir(parents=True)

                summary = {
                    "config": {"utility_kind": "smooth", "poster_min": 1, "poster_max": 4},
                    "inputs": {"utility_kind": "smooth"},
                    "data_summary": {"poster_min": 1, "poster_max": 4},
                    "metrics": {},
                }
                (variant_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

                U = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) / 10.0
                info = save_u_draw_archive(
                    idata=_FakeIdata(U),
                    out_path=archive_dir / "main_U_draws.npz",
                    posters=[1, 2, 3, 4],
                    U_true=[0.1, 0.2, 0.3, 0.4],
                    metadata={"scope": "main"},
                    dtype="float32",
                )
                write_archive_manifest(
                    archive_dir=archive_dir,
                    entries=[{"scope": "main", "kind": "u_draws", **info}],
                )

                old_plot = pp.plot_posterior_U_overlay_plotly
                pp.plot_posterior_U_overlay_plotly = _write_fake_plot
                try:
                    result = pp.postprocess_variant_dir(variant_dir, hdi_prob=0.5)
                finally:
                    pp.plot_posterior_U_overlay_plotly = old_plot

                self.assertEqual(result["status"], "updated")
                updated = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertTrue(updated["postprocess"]["updated_from_compact_archive"])
        finally:
            sampling.sample_model = old_sample_model


def _import_or_skip(module_name: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return __import__(module_name, fromlist=["*"])
    except Exception as exc:
        raise unittest.SkipTest(f"{module_name} unavailable in this environment: {exc}") from exc


def _raise_if_sampled(*args, **kwargs):
    raise AssertionError("sample_model must not be called during postprocess-only")


def _write_fake_plot(*args, **kwargs):
    out_path = Path(kwargs["out_path_html"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("<html></html>", encoding="utf-8")
    return None


if __name__ == "__main__":
    unittest.main()
