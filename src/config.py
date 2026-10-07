from dataclasses import dataclass, asdict

VALID_UTILITY_KINDS = ("smooth", "jagged")
VALID_UTILITY_KINDS_2D = ("smooth_contrast_interaction",)

@dataclass(frozen=True)
class MCMCConfig:
    draws: int = 1000
    tune: int = 1500
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class PrefixConfig:
    enabled: bool = False
    use_main_mcmc: bool = True
    mcmc: MCMCConfig = MCMCConfig(draws=400, tune=400, chains=2, cores=1)
    cache: bool = True
    overwrite_cache: bool = False   
    train_schedule: str = "geom2"
    train_geom_start: int = 8
    train_geom_ratio: int = 2
    train_k_traj_list: tuple[int, ...] = ()
    # Only used when train_schedule="fixed_step"
    train_traj_chunk_size: int = 300
    test_traj_chunk_size: int = 30


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = False
    utility_curve: bool = True
    rank_plots: bool = True
    prefix_heatmap: bool = False
    prefix_slider: bool = False


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool = False
    test_seen_path: str = ""
    test_unseen_path: str = ""
    obs_var_name: str = "obs"


@dataclass(frozen=True)
class PosteriorArchiveConfig:
    save_u_draws: bool = True
    dtype: str = "float32"
    save_diagnostics: bool = True
    save_full_idata: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    poster_min: int = 1
    poster_max: int = 100
    state_space: str = "1d"
    grid_k: int = 10
    
    utility_kind: str = "smooth" #"smooth" or "jagged"
    utility_metadata: dict | None = None
    
    beta_fixed: float = 20.0
    show_plot: bool = False
    model_name: str = "model2_discounted"
    alpha_fixed: float = 5.0
    alpha_prior_median: float = 5.0
    alpha_prior_log_sd: float = 0.75
    
    likelihood: str = "argmax"   # "softmax" or "argmax" or "boltzmann_qstar"
    prior: str = "rw"            # "rw" or "gp_rbf"

    # GP hyperpriors
    gp_eta_sd: float = 1.0
    gp_ls_sd: float = 0.25
    
    birl_export_mode: str = "stay" #or "terminal"
    birl_reward_scale: str = "discounted" #or "raw"
    birl_value_iters: int = 1000
    

    mcmc_main: MCMCConfig = MCMCConfig()
    prefix: PrefixConfig = PrefixConfig()
    plots: PlotConfig = PlotConfig()
    eval: EvalConfig = EvalConfig()
    posterior_archive: PosteriorArchiveConfig = PosteriorArchiveConfig()



def to_dict(cfg) -> dict:
    return asdict(cfg)


def validate_mcmc_config(m: MCMCConfig) -> None:
    for name in ("draws", "tune", "chains", "cores"):
        v = getattr(m, name)
        if not isinstance(v, int):
            raise TypeError(f"MCMCConfig.{name} must be int, got {type(v)}: {v}")
        if v <= 0:
            raise ValueError(f"MCMCConfig.{name} must be > 0, got {v}")

    if not isinstance(m.target_accept, (int, float)):
        raise TypeError(f"MCMCConfig.target_accept must be numeric, got {type(m.target_accept)}")
    if not (0 < float(m.target_accept) < 1):
        raise ValueError(f"MCMCConfig.target_accept must be in (0,1), got {m.target_accept}")


def validate_utility_kind(kind: str, *, state_space: str = "1d") -> str:
    k = str(kind).strip().lower()
    space = str(state_space).strip().lower()
    allowed = VALID_UTILITY_KINDS_2D if space == "2d" else VALID_UTILITY_KINDS
    if k not in allowed:
        raise ValueError(
            f"ExperimentConfig.utility_kind must be one of {allowed} "
            f"for state_space={space!r}, got {kind!r}."
        )
    return k


def validate_experiment_config(cfg: ExperimentConfig) -> None:
    state_space = str(getattr(cfg, "state_space", "1d")).strip().lower()
    if state_space not in {"1d", "2d"}:
        raise ValueError(f"state_space must be '1d' or '2d', got {state_space!r}.")

    grid_k = int(getattr(cfg, "grid_k", 10))
    if grid_k <= 0:
        raise ValueError(f"grid_k must be positive, got {grid_k}.")
    if state_space == "2d" and grid_k * grid_k != int(cfg.poster_max - cfg.poster_min + 1):
        raise ValueError(
            "For state_space='2d', poster_min/poster_max must span grid_k^2 states; "
            f"got grid_k={grid_k}, poster range [{cfg.poster_min},{cfg.poster_max}]."
        )

    validate_utility_kind(str(getattr(cfg, "utility_kind", "smooth")), state_space=state_space)

    model_name = str(cfg.model_name).strip().lower()
    valid_models = {
        "model2_discounted",
        "model4_endpoint_discounted_boltzmann",
        "model4_endpoint_discounted_boltzmann_infer_alpha",
        "birl_baseline_discounted_boltzmann",
        "birl_baseline",
        "pbo_baseline",
        "pbo_baseline_start_end_preference",
        "pbo_baseline_transition_preference",
    }
    if model_name not in valid_models:
        raise ValueError(
            f"Unknown model_name={cfg.model_name!r}. "
            f"Expected one of {sorted(valid_models)}."
        )

    if model_name == "model4_endpoint_discounted_boltzmann_infer_alpha" and state_space != "1d":
        raise ValueError(
            "model4_endpoint_discounted_boltzmann_infer_alpha is v1 1D-only; "
            f"got state_space={state_space!r}."
        )

    alpha_prior_median = float(getattr(cfg, "alpha_prior_median", 5.0))
    if alpha_prior_median <= 0.0:
        raise ValueError(
            f"alpha_prior_median must be positive, got {alpha_prior_median}."
        )

    alpha_prior_log_sd = float(getattr(cfg, "alpha_prior_log_sd", 0.75))
    if alpha_prior_log_sd <= 0.0:
        raise ValueError(
            f"alpha_prior_log_sd must be positive, got {alpha_prior_log_sd}."
        )

    export_mode = str(getattr(cfg, "birl_export_mode", "stay")).strip().lower()
    if export_mode not in {"stay", "terminal"}:
        raise ValueError(
            f"birl_export_mode must be 'stay' or 'terminal', got {export_mode!r}."
        )

    reward_scale = str(getattr(cfg, "birl_reward_scale", "discounted")).strip().lower()
    if reward_scale not in {"discounted", "scaled", "one_minus_gamma", "raw", "unscaled", "none"}:
        raise ValueError(
            "birl_reward_scale must be one of "
            "'discounted', 'scaled', 'one_minus_gamma', 'raw', 'unscaled', or 'none'. "
            f"Got {reward_scale!r}."
        )

    value_iters = int(getattr(cfg, "birl_value_iters", 1000))
    if value_iters <= 0:
        raise ValueError(f"birl_value_iters must be positive, got {value_iters}.")

    archive = getattr(cfg, "posterior_archive", PosteriorArchiveConfig())
    dtype = str(getattr(archive, "dtype", "float32")).strip()
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"posterior_archive.dtype must be 'float32' or 'float64', got {dtype!r}.")
