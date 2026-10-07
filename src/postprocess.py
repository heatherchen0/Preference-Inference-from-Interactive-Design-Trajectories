from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .posterior_archive import load_u_draw_archive, read_archive_manifest
from .posterior_archive import write_json as save_json
from .metrics import (
    summarize_draws_hdi,
    normalize_draws_minmax,
    compute_curve_error_metrics,
    compute_gaussian_log_score_metrics,
    compute_pairwise_accuracy,
    compute_posterior_pairwise_accuracy,
    soft_copeland_draws,
    compute_recommendation_metrics,
    compute_contrast_profile_metrics,
    build_contrast_profile_table,
    write_contrast_profile_table,
)
from .prefix_analysis import (
    materialize_prefix_summaries,
    write_prefix_metrics_csv,
    write_prefix_per_state_tables,
)
from .plots import (
    plot_posterior_U_overlay_plotly,
    plot_interactive_prefix_posterior_states_plotly,
    plot_prefix_mean_heatmap_states_plotly,
    plot_prefix_metrics_learning_curves_plotly,
    plot_2d_posterior_heatmaps_plotly,
    plot_2d_prefix_heatmap_slider_plotly,
    plot_contrast_profile_plotly,
)
from utility import get_utility
from .state_space import normalize_state_space


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_artifact_path(path_value, *, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    p = Path(str(path_value))
    if p.exists():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    return p


def _utility_from_summary(summary: dict):
    inputs = summary.get("inputs", {}) if isinstance(summary.get("inputs", {}), dict) else {}
    config = summary.get("config", {}) if isinstance(summary.get("config", {}), dict) else {}
    state_space = normalize_state_space(inputs.get("state_space", config.get("state_space", "1d")))
    utility_kind = str(inputs.get("utility_kind", config.get("utility_kind", "smooth"))).strip().lower()
    grid_k = inputs.get("grid_k", config.get("grid_k", None))

    if state_space == "2d":
        if utility_kind != "smooth_contrast_interaction":
            raise ValueError(f"Unsupported 2D utility_kind={utility_kind!r}.")
        from poster2d.utility import smooth_contrast_interaction_utility

        k = int(grid_k)

        def utility_fn(posters):
            return smooth_contrast_interaction_utility(posters, k=k)

        return utility_fn

    return get_utility(utility_kind)


def _postprocess_geometry_from_summary(summary: dict) -> tuple[str, int | None, float | None]:
    inputs = summary.get("inputs", {}) if isinstance(summary.get("inputs", {}), dict) else {}
    config = summary.get("config", {}) if isinstance(summary.get("config", {}), dict) else {}
    state_space = normalize_state_space(inputs.get("state_space", config.get("state_space", "1d")))
    grid_k = inputs.get("grid_k", config.get("grid_k", None))
    grid_k = None if grid_k is None else int(grid_k)
    preferred_contrast = inputs.get("preferred_contrast", None)
    if preferred_contrast is None:
        meta = inputs.get("utility_metadata", config.get("utility_metadata", {}))
        if isinstance(meta, dict):
            preferred_contrast = meta.get("preferred_contrast", None)
    preferred_contrast = None if preferred_contrast is None else float(preferred_contrast)
    return state_space, grid_k, preferred_contrast


def _find_archive_entry(manifest: dict, *, scope: str, kind: str = "u_draws", k_traj: int | None = None):
    for entry in manifest.get("entries", []) or []:
        if str(entry.get("scope")) != str(scope):
            continue
        if str(entry.get("kind", "u_draws")) != str(kind):
            continue
        if k_traj is not None and int(entry.get("k_traj", -1)) != int(k_traj):
            continue
        return entry
    return None


def _mark_archive_unavailable(summary: dict, summary_path: Path, *, status: str) -> None:
    summary.setdefault("posterior_archive", {})["postprocess_status"] = str(status)
    summary.setdefault("postprocess", {})["updated_from_compact_archive"] = False
    summary["postprocess"]["posterior_sample_recomputations"] = "unavailable"
    summary["postprocess"]["unavailable_reason"] = str(status)
    summary["postprocess"]["legacy_summary_available"] = True
    save_json(summary, summary_path)


def _summaries_and_metrics_from_draws(
    *,
    draws_chain: np.ndarray,
    posters: np.ndarray,
    U_true: np.ndarray,
    hdi_prob: float,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
) -> dict:
    state_space = normalize_state_space(state_space)
    draws_chain = np.asarray(draws_chain, dtype=float)
    flat = draws_chain.reshape((draws_chain.shape[0] * draws_chain.shape[1], draws_chain.shape[2]))

    raw = summarize_draws_hdi(flat, hdi_prob=float(hdi_prob))
    norm_draws, norm_info = normalize_draws_minmax(flat)
    norm = summarize_draws_hdi(norm_draws, hdi_prob=float(hdi_prob))
    norm["normalization"] = norm_info

    raw_metrics = compute_curve_error_metrics(
        posters=posters,
        U_true=U_true,
        U_post_mean=np.asarray(raw["mean"], dtype=float),
        U_low=np.asarray(raw["low"], dtype=float),
        U_high=np.asarray(raw["high"], dtype=float),
    )
    log_score_raw = compute_gaussian_log_score_metrics(
        U_true=U_true,
        U_post_mean=np.asarray(raw["mean"], dtype=float),
        U_post_sd=np.asarray(raw["sd"], dtype=float),
        variance_floor=1e-6,
    )
    raw_metrics = {
        **raw_metrics,
        "gaussian_nlpd_raw": log_score_raw["gaussian_nlpd"],
        "gaussian_log_score_raw": log_score_raw["gaussian_log_score"],
        "gaussian_nlpd_var_floor": log_score_raw["variance_floor"],
        "gaussian_nlpd_num_floored": log_score_raw["num_floored"],
        "gaussian_nlpd_fraction_floored": log_score_raw["fraction_floored"],
        "gaussian_nlpd_n_points": log_score_raw["n_points"],
    }

    norm_metrics = compute_curve_error_metrics(
        posters=posters,
        U_true=U_true,
        U_post_mean=np.asarray(norm["mean"], dtype=float),
        U_low=np.asarray(norm["low"], dtype=float),
        U_high=np.asarray(norm["high"], dtype=float),
    )

    ranking_raw = compute_pairwise_accuracy(
        U_true=U_true,
        U_pred=np.asarray(raw["mean"], dtype=float),
        eps=1e-6,
    )
    ranking_norm = compute_pairwise_accuracy(
        U_true=U_true,
        U_pred=np.asarray(norm["mean"], dtype=float),
        eps=1e-6,
    )
    ranking_posterior = compute_posterior_pairwise_accuracy(
        U_true=U_true,
        U_draws=flat,
        eps_true=1e-6,
        eps_pred=0.0,
    )

    soft_copeland = summarize_draws_hdi(
        soft_copeland_draws(flat),
        hdi_prob=float(hdi_prob),
    )
    recommendation = compute_recommendation_metrics(
        posters=posters,
        U_true=U_true,
        recommendation_score=np.asarray(soft_copeland["mean"], dtype=float),
        score_name="soft_copeland",
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )

    norm_metrics = {
        **norm_metrics,
        "pairwise_acc_comp": ranking_norm["pairwise_acc_comp"],
        "pairwise_acc_strict": ranking_norm["pairwise_acc_strict"],
        "pairwise_num_pairs_comparable": ranking_norm["num_pairs_comparable"],
        "pairwise_num_correct_comparable": ranking_norm["num_correct_comparable"],
        "pairwise_eps": ranking_norm["eps"],
        "posterior_pairwise_acc_comp": ranking_posterior["posterior_pairwise_acc_comp"],
        "posterior_pairwise_acc_strict": ranking_posterior["posterior_pairwise_acc_strict"],
        "posterior_pairwise_num_pairs_comparable": ranking_posterior["num_pairs_comparable"],
        "posterior_pairwise_expected_correct_comparable": ranking_posterior[
            "posterior_pairwise_expected_correct_comparable"
        ],
        "posterior_pairwise_num_draws": ranking_posterior["n_draws"],
        "posterior_pairwise_eps_true": ranking_posterior["eps_true"],
        "posterior_pairwise_eps_pred": ranking_posterior["eps_pred"],
        "posterior_pairwise_acc_draw_sd": ranking_posterior["posterior_pairwise_acc_draw_sd"],
        "normalization_method": "posterior_draw_minmax",
        "g_hat": recommendation["g_hat"],
        "U_true_at_g_hat": recommendation["U_true_at_g_hat"],
        "simple_regret": recommendation["simple_regret"],
        "g_star": recommendation["g_star"],
        "U_true_star": recommendation["U_true_star"],
    }
    if state_space == "2d":
        norm_metrics.update(
            compute_contrast_profile_metrics(
                U_true=U_true,
                U_pred=np.asarray(norm["mean"], dtype=float),
                grid_k=int(grid_k),
            )
        )
        norm_metrics.update(
            {
                "h_hat": recommendation.get("h_hat"),
                "b_hat": recommendation.get("b_hat"),
                "c_hat": recommendation.get("c_hat"),
                "h_star": recommendation.get("h_star"),
                "b_star": recommendation.get("b_star"),
                "c_star": recommendation.get("c_star"),
                "contrast_error": recommendation.get("contrast_error"),
                "signed_contrast_error": recommendation.get("signed_contrast_error"),
            }
        )

    return {
        "raw": raw,
        "norm": norm,
        "metrics_raw": raw_metrics,
        "metrics_norm": norm_metrics,
        "ranking_raw": ranking_raw,
        "ranking_norm": ranking_norm,
        "ranking_posterior": ranking_posterior,
        "soft_copeland": soft_copeland,
        "recommendation": recommendation,
    }


def _update_prefix_from_archive(
    prefix: dict,
    *,
    archive_path: Path,
    hdi_prob: float,
    state_space: str = "1d",
    grid_k: int | None = None,
    preferred_contrast: int | float | None = None,
) -> dict:
    loaded = load_u_draw_archive(archive_path)
    posters = np.asarray(loaded["posters"], dtype=int)
    U_true = np.asarray(loaded["U_true"], dtype=float)
    bundle = _summaries_and_metrics_from_draws(
        draws_chain=np.asarray(loaded["U"]),
        posters=posters,
        U_true=U_true,
        hdi_prob=float(hdi_prob),
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )

    out = dict(prefix)
    out.update(
        {
            "mean": np.asarray(bundle["raw"]["mean"], dtype=float),
            "sd": np.asarray(bundle["raw"]["sd"], dtype=float),
            "low": np.asarray(bundle["raw"]["low"], dtype=float),
            "high": np.asarray(bundle["raw"]["high"], dtype=float),
            "mean_norm_draws": np.asarray(bundle["norm"]["mean"], dtype=float),
            "sd_norm_draws": np.asarray(bundle["norm"]["sd"], dtype=float),
            "low_norm_draws": np.asarray(bundle["norm"]["low"], dtype=float),
            "high_norm_draws": np.asarray(bundle["norm"]["high"], dtype=float),
            "normalization_norm_draws": bundle["norm"]["normalization"],
            "soft_copeland_mean": np.asarray(bundle["soft_copeland"]["mean"], dtype=float),
            "soft_copeland_sd": np.asarray(bundle["soft_copeland"]["sd"], dtype=float),
            "soft_copeland_low": np.asarray(bundle["soft_copeland"]["low"], dtype=float),
            "soft_copeland_high": np.asarray(bundle["soft_copeland"]["high"], dtype=float),
            "soft_copeland_preference_scale": 1.0,
            "posterior_pairwise_acc_comp": bundle["ranking_posterior"]["posterior_pairwise_acc_comp"],
            "posterior_pairwise_acc_strict": bundle["ranking_posterior"]["posterior_pairwise_acc_strict"],
            "posterior_pairwise_num_pairs_comparable": bundle["ranking_posterior"]["num_pairs_comparable"],
            "posterior_pairwise_expected_correct_comparable": bundle["ranking_posterior"][
                "posterior_pairwise_expected_correct_comparable"
            ],
            "posterior_pairwise_num_draws": bundle["ranking_posterior"]["n_draws"],
            "posterior_pairwise_eps_true": bundle["ranking_posterior"]["eps_true"],
            "posterior_pairwise_eps_pred": bundle["ranking_posterior"]["eps_pred"],
            "posterior_pairwise_acc_draw_sd": bundle["ranking_posterior"][
                "posterior_pairwise_acc_draw_sd"
            ],
            "postprocessed_from_archive": True,
        }
    )
    return out


def postprocess_variant_dir(variant_dir: str | Path, *, hdi_prob: float = 0.94) -> dict:
    variant_dir = Path(variant_dir)
    summary_path = variant_dir / "summary.json"
    if not summary_path.exists():
        return {"variant_dir": str(variant_dir), "status": "missing_summary"}

    summary = _read_json(summary_path)
    state_space, grid_k, preferred_contrast = _postprocess_geometry_from_summary(summary)
    archive_dir = variant_dir / "posterior_archive"
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.exists():
        _mark_archive_unavailable(summary, summary_path, status="missing_archive")
        return {"variant_dir": str(variant_dir), "status": "missing_archive"}

    manifest = read_archive_manifest(manifest_path)
    main_entry = _find_archive_entry(manifest, scope="main", kind="u_draws")
    if main_entry is None:
        _mark_archive_unavailable(summary, summary_path, status="missing_main_u_draws")
        return {"variant_dir": str(variant_dir), "status": "missing_main_u_draws"}

    main_path = _resolve_artifact_path(main_entry.get("path"), base_dir=variant_dir)
    if main_path is None or not main_path.exists():
        _mark_archive_unavailable(summary, summary_path, status="missing_main_u_draws_file")
        return {"variant_dir": str(variant_dir), "status": "missing_main_u_draws_file"}

    loaded = load_u_draw_archive(main_path)
    posters = np.asarray(loaded["posters"], dtype=int)
    U_true = np.asarray(loaded["U_true"], dtype=float)
    bundle = _summaries_and_metrics_from_draws(
        draws_chain=np.asarray(loaded["U"]),
        posters=posters,
        U_true=U_true,
        hdi_prob=float(hdi_prob),
        state_space=state_space,
        grid_k=grid_k,
        preferred_contrast=preferred_contrast,
    )

    summary["metrics"]["raw_mse_U_mean_vs_ground_truth"] = bundle["metrics_raw"]["mse"]
    summary["metrics"]["curve_error"] = bundle["metrics_norm"]
    summary["metrics"]["curve_error_posterior_draw_minmax"] = bundle["metrics_norm"]
    summary["metrics"]["curve_error_raw"] = bundle["metrics_raw"]
    summary["metrics"]["log_score_raw"] = {
        "gaussian_nlpd_raw": bundle["metrics_raw"]["gaussian_nlpd_raw"],
        "gaussian_log_score_raw": bundle["metrics_raw"]["gaussian_log_score_raw"],
        "gaussian_nlpd_var_floor": bundle["metrics_raw"]["gaussian_nlpd_var_floor"],
        "gaussian_nlpd_num_floored": bundle["metrics_raw"]["gaussian_nlpd_num_floored"],
    }
    summary["metrics"]["ranking"] = bundle["ranking_norm"]
    summary["metrics"]["ranking_raw"] = bundle["ranking_raw"]
    summary["metrics"]["ranking_posterior"] = bundle["ranking_posterior"]
    summary["metrics"]["recommendation"] = bundle["recommendation"]
    summary["curves"] = {
        "poster": posters,
        "U_true": U_true,
        "U_post_mean": np.asarray(bundle["raw"]["mean"], dtype=float),
        "U_post_sd": np.asarray(bundle["raw"]["sd"], dtype=float),
    }
    summary["U_summary"] = {
        "mean": np.asarray(bundle["raw"]["mean"], dtype=float),
        "sd": np.asarray(bundle["raw"]["sd"], dtype=float),
        "low": np.asarray(bundle["raw"]["low"], dtype=float),
        "high": np.asarray(bundle["raw"]["high"], dtype=float),
    }
    summary["U_summary_norm_draws"] = {
        "mean": np.asarray(bundle["norm"]["mean"], dtype=float),
        "sd": np.asarray(bundle["norm"]["sd"], dtype=float),
        "low": np.asarray(bundle["norm"]["low"], dtype=float),
        "high": np.asarray(bundle["norm"]["high"], dtype=float),
        "normalization": bundle["norm"]["normalization"],
    }
    summary["soft_copeland_summary"] = {
        "mean": np.asarray(bundle["soft_copeland"]["mean"], dtype=float),
        "sd": np.asarray(bundle["soft_copeland"]["sd"], dtype=float),
        "low": np.asarray(bundle["soft_copeland"]["low"], dtype=float),
        "high": np.asarray(bundle["soft_copeland"]["high"], dtype=float),
        "preference_scale": 1.0,
    }
    summary.setdefault("posterior_archive", {})["postprocess_status"] = "updated_from_archive"
    summary["posterior_archive"]["manifest_json"] = str(manifest_path)

    tables_dir = variant_dir / "tables"
    from .metrics import write_per_state_utility_table

    write_per_state_utility_table(
        out_path=tables_dir / "utility_per_state_raw.csv",
        posters=posters,
        U_true=U_true,
        U_post_mean=np.asarray(bundle["raw"]["mean"], dtype=float),
        U_post_sd=np.asarray(bundle["raw"]["sd"], dtype=float),
        U_low=np.asarray(bundle["raw"]["low"], dtype=float),
        U_high=np.asarray(bundle["raw"]["high"], dtype=float),
    )
    write_per_state_utility_table(
        out_path=tables_dir / "utility_per_state_posterior_draw_minmax.csv",
        posters=posters,
        U_true=U_true,
        U_post_mean=np.asarray(bundle["norm"]["mean"], dtype=float),
        U_post_sd=np.asarray(bundle["norm"]["sd"], dtype=float),
        U_low=np.asarray(bundle["norm"]["low"], dtype=float),
        U_high=np.asarray(bundle["norm"]["high"], dtype=float),
    )
    if state_space == "2d":
        write_contrast_profile_table(
            out_path=tables_dir / "contrast_profile_raw.csv",
            U_true=U_true,
            U_post_mean=np.asarray(bundle["raw"]["mean"], dtype=float),
            U_post_sd=np.asarray(bundle["raw"]["sd"], dtype=float),
            U_low=np.asarray(bundle["raw"]["low"], dtype=float),
            U_high=np.asarray(bundle["raw"]["high"], dtype=float),
            grid_k=int(grid_k),
        )
        write_contrast_profile_table(
            out_path=tables_dir / "contrast_profile_posterior_draw_minmax.csv",
            U_true=U_true,
            U_post_mean=np.asarray(bundle["norm"]["mean"], dtype=float),
            U_post_sd=np.asarray(bundle["norm"]["sd"], dtype=float),
            U_low=np.asarray(bundle["norm"]["low"], dtype=float),
            U_high=np.asarray(bundle["norm"]["high"], dtype=float),
            grid_k=int(grid_k),
        )

    utility_kind = str(summary.get("inputs", {}).get("utility_kind", summary.get("config", {}).get("utility_kind", "smooth")))
    poster_min = int(summary.get("data_summary", {}).get("poster_min", int(posters[0])))
    poster_max = int(summary.get("data_summary", {}).get("poster_max", int(posters[-1])))
    plots_dir = variant_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    main_plot_outputs = {}
    raw_plot_path = plots_dir / "posterior_utility_curve_raw.html"
    plot_posterior_U_overlay_plotly(
        [
            {
                "label": "posterior",
                "mean": np.asarray(bundle["raw"]["mean"], dtype=float),
                "low": np.asarray(bundle["raw"]["low"], dtype=float),
                "high": np.asarray(bundle["raw"]["high"], dtype=float),
            }
        ],
        poster_min=poster_min,
        poster_max=poster_max,
        out_path_html=raw_plot_path,
        title="Posterior utility curve, raw U(g), postprocessed",
        scale_mode="raw",
        yaxis_title="Utility U(g)",
        hdi_prob=float(hdi_prob),
        show_hdi_band=True,
        truth_y=U_true,
        truth_label=f"Ground truth U(g), {utility_kind}",
        truth_color="black",
        truth_dash="solid",
        truth_width=4,
    )
    main_plot_outputs["U_band_raw_html"] = str(raw_plot_path)

    minmax_plot_path = plots_dir / "posterior_utility_curve.html"
    plot_posterior_U_overlay_plotly(
        [
            {
                "label": "posterior",
                "mean": np.asarray(bundle["norm"]["mean"], dtype=float),
                "low": np.asarray(bundle["norm"]["low"], dtype=float),
                "high": np.asarray(bundle["norm"]["high"], dtype=float),
                "already_normalized": True,
                "normalization_method": "posterior_draw_minmax",
            }
        ],
        poster_min=poster_min,
        poster_max=poster_max,
        out_path_html=minmax_plot_path,
        title="Posterior utility curve, posterior-draw min-max normalized, postprocessed",
        scale_mode="posterior_draw_minmax",
        yaxis_title="Min-max normalized utility",
        hdi_prob=float(hdi_prob),
        show_hdi_band=True,
        truth_y=U_true,
        truth_label=f"Ground truth U(g), {utility_kind}",
        truth_color="black",
        truth_dash="solid",
        truth_width=4,
    )
    main_plot_outputs["U_band_minmax_html"] = str(minmax_plot_path)

    if state_space == "2d":
        heatmap_path = plots_dir / "posterior_utility_heatmaps_2d.html"
        plot_2d_posterior_heatmaps_plotly(
            U_true=U_true,
            U_post_mean=np.asarray(bundle["norm"]["mean"], dtype=float),
            U_post_sd=np.asarray(bundle["norm"]["sd"], dtype=float),
            grid_k=int(grid_k),
            out_path_html=heatmap_path,
            title="2D posterior utility heatmaps, postprocessed",
            recommendation=bundle["recommendation"],
            zmin=0.0,
            zmax=1.0,
        )
        main_plot_outputs["posterior_utility_heatmaps_2d_html"] = str(heatmap_path)

        contrast_path = plots_dir / "contrast_profile_posterior_draw_minmax.html"
        plot_contrast_profile_plotly(
            build_contrast_profile_table(
                U_true=U_true,
                U_post_mean=np.asarray(bundle["norm"]["mean"], dtype=float),
                U_low=np.asarray(bundle["norm"]["low"], dtype=float),
                U_high=np.asarray(bundle["norm"]["high"], dtype=float),
                grid_k=int(grid_k),
            ),
            out_path_html=contrast_path,
            title="Contrast profile recovery, postprocessed",
            preferred_contrast=preferred_contrast,
        )
        main_plot_outputs["contrast_profile_minmax_html"] = str(contrast_path)

    prefix_cache_path = variant_dir / "prefix_summaries.json"
    prefix_summaries = []
    if prefix_cache_path.exists():
        prefix_cache = _read_json(prefix_cache_path)
        for prefix in prefix_cache.get("prefix_summaries", []) or []:
            archive_path = _resolve_artifact_path(
                prefix.get("posterior_archive_u_draws_path"),
                base_dir=variant_dir,
            )
            if archive_path is None:
                entry = _find_archive_entry(
                    manifest,
                    scope="prefix",
                    kind="u_draws",
                    k_traj=int(prefix.get("k_traj", -1)),
                )
                archive_path = _resolve_artifact_path(entry.get("path"), base_dir=variant_dir) if entry else None
            if archive_path is not None and archive_path.exists():
                prefix_summaries.append(
                    _update_prefix_from_archive(
                        prefix,
                        archive_path=archive_path,
                        hdi_prob=float(hdi_prob),
                        state_space=state_space,
                        grid_k=grid_k,
                        preferred_contrast=preferred_contrast,
                    )
                )
            else:
                prefix_summaries.append(dict(prefix))

    prefix_outputs = {}
    if prefix_summaries:
        utility_fn = _utility_from_summary(summary)
        prefix_mat = materialize_prefix_summaries(
            prefix_summaries,
            utility_fn=utility_fn,
            utility_kind=utility_kind,
            poster_min=poster_min,
            poster_max=poster_max,
            hdi_prob=float(hdi_prob),
            state_space=state_space,
            grid_k=grid_k,
            preferred_contrast=preferred_contrast,
        )
        save_json({"prefix_summaries": prefix_mat}, variant_dir / "prefix_summaries.materialized.json")
        write_prefix_metrics_csv(prefix_mat, tables_dir / "prefix_metrics_minmax.csv")
        write_prefix_per_state_tables(prefix_mat, out_dir=tables_dir / "prefix_tables")
        if state_space == "2d":
            from .prefix_analysis import write_prefix_contrast_profile_tables

            write_prefix_contrast_profile_tables(
                prefix_mat,
                out_dir=tables_dir / "prefix_contrast_profiles",
                grid_k=int(grid_k),
            )
        prefix_outputs["prefix_summaries_json"] = str(variant_dir / "prefix_summaries.materialized.json")

        prefix_for_plots = []
        prefix_for_raw_plots = []
        for s in prefix_mat:
            c = s["curves_minmax"]
            s_plot = dict(s)
            s_plot["mean"] = np.asarray(c["U_post_mean"], dtype=float)
            s_plot["low"] = np.asarray(c["U_hdi_low"], dtype=float)
            s_plot["high"] = np.asarray(c["U_hdi_high"], dtype=float)
            prefix_for_plots.append(s_plot)

            c_raw = s["curves"]
            s_raw_plot = dict(s)
            s_raw_plot["mean"] = np.asarray(c_raw["U_post_mean"], dtype=float)
            s_raw_plot["low"] = np.asarray(c_raw["U_hdi_low"], dtype=float)
            s_raw_plot["high"] = np.asarray(c_raw["U_hdi_high"], dtype=float)
            prefix_for_raw_plots.append(s_raw_plot)

        plot_interactive_prefix_posterior_states_plotly(
            prefix_for_plots,
            poster_min=poster_min,
            poster_max=poster_max,
            out_path_html=plots_dir / "posterior_utility_curve_slider.html",
            title="Posterior of min-max normalized U(g), postprocessed",
            yaxis_title="Min-max normalized utility",
            yaxis_range=[0.0, 1.0],
        )
        plot_interactive_prefix_posterior_states_plotly(
            prefix_for_raw_plots,
            poster_min=poster_min,
            poster_max=poster_max,
            out_path_html=plots_dir / "posterior_utility_curve_slider_raw.html",
            title="Posterior of raw U(g), postprocessed",
            yaxis_title="Utility U(g)",
        )
        plot_prefix_metrics_learning_curves_plotly(
            prefix_mat,
            out_path_html=plots_dir / "rmse_interval_coverage.html",
            title="Prefix metrics, postprocessed",
            use_metrics="minmax",
        )
        plot_prefix_mean_heatmap_states_plotly(
            prefix_for_plots,
            poster_min=poster_min,
            poster_max=poster_max,
            out_path_html=plots_dir / "posterior_mean_heatmap.html",
            title="Heatmap of min-max normalized posterior mean U(g), postprocessed",
            zmin=0.0,
            zmax=1.0,
        )
        if state_space == "2d":
            plot_2d_prefix_heatmap_slider_plotly(
                prefix_for_plots,
                grid_k=int(grid_k),
                out_path_html=plots_dir / "posterior_utility_heatmap_slider_2d.html",
                title="2D posterior mean utility by prefix, postprocessed",
                value_key="mean",
                zmin=0.0,
                zmax=1.0,
            )

    summary.setdefault("postprocess", {})["updated_from_compact_archive"] = True
    summary["postprocess"]["hdi_prob"] = float(hdi_prob)
    summary["postprocess"]["main_plot_outputs"] = main_plot_outputs
    summary["postprocess"]["prefix_outputs"] = prefix_outputs
    save_json(summary, summary_path)

    return {"variant_dir": str(variant_dir), "status": "updated", **main_plot_outputs, **prefix_outputs}


def postprocess_run_root(run_root: str | Path, *, utility_mode: str = "both", hdi_prob: float = 0.94) -> list[dict]:
    run_root = Path(run_root)
    results = []
    for utility_dir in sorted(run_root.glob("utility=*")):
        if not utility_dir.is_dir():
            continue
        utility_kind = utility_dir.name.split("=", 1)[1].strip().lower()
        if utility_mode != "both" and utility_kind != utility_mode:
            continue
        for dataset_dir in sorted(p for p in utility_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                if not seed_dir.name.startswith("seed="):
                    continue
                for variant_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("v=")):
                    results.append(postprocess_variant_dir(variant_dir, hdi_prob=float(hdi_prob)))
    return results
