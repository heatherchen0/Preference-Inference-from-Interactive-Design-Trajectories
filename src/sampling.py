import pymc as pm
from .config import MCMCConfig, validate_mcmc_config


def sample_model(
    model: pm.Model,
    mcmc: MCMCConfig | None = None,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
):
    if mcmc is None:
        mcmc = MCMCConfig(
            draws=1000 if draws is None else int(draws),
            tune=1500 if tune is None else int(tune),
            chains=4 if chains is None else int(chains),
            cores=1 if cores is None else int(cores),
            target_accept=0.95 if target_accept is None else float(target_accept),
            random_seed=42 if random_seed is None else int(random_seed),
        )

    validate_mcmc_config(mcmc)

    with model:
        idata = pm.sample(
            draws=mcmc.draws,
            tune=mcmc.tune,
            chains=mcmc.chains,
            cores=mcmc.cores,
            target_accept=mcmc.target_accept,
            random_seed=mcmc.random_seed,
            return_inferencedata=True,
        )
    return idata