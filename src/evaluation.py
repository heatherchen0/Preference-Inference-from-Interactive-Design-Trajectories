from pathlib import Path
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def summarize_posterior(idata, var_names=None, hdi_prob=0.94):
    if var_names is None:
        var_names = [v for v in list(idata.posterior.data_vars) if v not in ("probs",)]
    return az.summary(idata, var_names=var_names, hdi_prob=hdi_prob)


def compute_diagnostics_summary(idata):
    ess = az.ess(idata)
    rhat = az.rhat(idata)
    out = {
        "ess": {k: float(np.nanmin(ess[k].values)) for k in ess.data_vars},
        "rhat": {k: float(np.nanmax(rhat[k].values)) for k in rhat.data_vars},
    }
    return out


def save_posterior_summaries(summary_df, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(path, index=True)


def save_chain_rank_plots(idata, run_dir: Path, var_names=None, max_vars=12):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if var_names is None:
        var_names = [v for v in idata.posterior.data_vars if v not in ("probs",)]
        var_names = var_names[:max_vars]

    for v in var_names:
        try:
            ax = az.plot_rank(idata, var_names=[v], kind="vlines")
            fig = ax.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(run_dir / f"rank_{v}.png", dpi=200)
            plt.close(fig)
        except Exception as e:
            (run_dir / f"rank_{v}_ERROR.txt").write_text(str(e), encoding="utf-8")