# Preference Inference from Interactive Design Trajectories

Research code for inferring a user's hidden preference over design states from their sequence of edits and final export action. The project uses simulated grayscale poster editors to study how much preference information can be recovered from interaction trajectories.

The repository includes a one-dimensional editor, a two-dimensional extension, Bayesian inference with PyMC, comparison baselines, and evaluation of utility recovery and recommendation quality. The bundled datasets are synthetic.

## Methods

- **Endpoint-discounted model (Model 4):** models edit and export actions using discounted endpoint utility and a Boltzmann action likelihood.
- **BIRL-style baseline:** infers utility using a discounted state-reward model and a Boltzmann action likelihood.
- **Start-end preference baseline:** treats the exported state as preferred to the trajectory's starting state.
- **Transition preference baseline:** treats each observed next state as preferred to its preceding state.

The official 1D comparison uses both a Gaussian-process prior and an independent (i.i.d.) prior for each method, giving eight variants. Evaluation includes utility RMSE, posterior pairwise ranking accuracy, simple regret, interval coverage, and raw-scale Gaussian negative log predictive density. An inferred-temperature extension of Model 4 is also implemented.

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution, then run these commands from a terminal with Conda available. On Windows, an Anaconda or Miniforge Prompt is suitable.

```sh
git clone https://github.com/heatherchen0/Preference-Inference-from-Interactive-Design-Trajectories.git
cd Preference-Inference-from-Interactive-Design-Trajectories
conda env create --file environment.yml
conda activate preference-inference
```

If already cloned the repository, start from its root and run the two Conda commands. Run all commands below from the repository root with this environment active.

[environment.yml](environment.yml) uses Python 3.11 and the recorded thesis versions of the main numerical packages, including PyMC 5.28.1. It also includes Plotly for HTML figures and h5netcdf for optional full posterior archives. It contains no local installation path or operating-system build strings; Conda resolves platform-specific and transitive dependencies. It is a portable environment specification, not a complete lockfile.

See the [Conda environment documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) for environment setup and activation help.

## Quick checks

First verify that the inference dependencies import, then run the tests:

```sh
python -c "import numpy, pandas, scipy, pymc, pytensor, arviz, matplotlib, plotly, h5netcdf"
python -B -m unittest discover -s tests -v
python -B -m poster2d.smoke_test
```

Some tests skip when an inference dependency cannot be imported. Check the import command and the test summary; skipped tests do not establish that the full inference stack works. Tests use Python's standard-library `unittest`, so pytest is not required.

Validate one official 1D task without sampling or writing results:

```sh
python triton_run_main_candidate_matched_task.py --task-csv triton_main_1d_official_seed0-9_tasks.csv --task-id 5 --run-root results/main_1d --dry-run
```

This prints the configuration for the smooth-utility endpoint model with a GP prior, seed 0, gamma 0.97, and alpha 10. The script runs locally despite its `triton_` name; a cluster or SLURM installation is not required.

## Official 1D experiments

### Included data

The design state is a grayscale value `g` in `{1, ..., 100}`. The available actions are `dec`, `export`, and `inc`; export terminates a trajectory.

The two datasets under [data/main_1d/](data/main_1d/) use the endpoint-discounted simulator with **gamma = 0.97**, **alpha = 10**, and data-generation seed **0**. Each contains **500 trajectories**: five repetitions from each of the 100 starting states, with trajectory order shuffled.

| Utility | File under `data/main_1d/` | Step records |
| --- | --- | ---: |
| Smooth | `steps_UserModel4Boltzmann_gamma0.97_alpha10_reps5_uniform_train.jsonl` | 7,797 |
| Jagged | `steps_Jagged_UserModel4Boltzmann_gamma0.97_alpha10_reps5_uniform_train.jsonl` | 6,884 |

Each JSONL line is one observed state/action pair. For example:

```json
{"traj_id": 0, "t": 0, "poster": 51, "action": "dec", "start_poster": 51, "rep": 2, "model": "UserModel4Boltzmann_gamma0.97_alpha10_reps5_uniform_train"}
```

The core observation fields are `traj_id`, `t`, `poster`, and `action`. Rows within a trajectory are ordered by `t`; the terminal row records an `export` action. The remaining fields describe the simulation provenance.

### Run a single task

The supplied [task CSV](triton_main_1d_official_seed0-9_tasks.csv) defines 160 independent tasks: two utilities, eight method/prior variants, and ten inference seeds. The seeds change trajectory ordering and MCMC initialization for the same two bundled datasets; they are not ten independently generated datasets.

To run the endpoint model with the GP prior on smooth utility, use task 5:

```sh
python triton_run_main_candidate_matched_task.py --task-csv triton_main_1d_official_seed0-9_tasks.csv --task-id 5 --run-root results/main_1d --cores 1
```

Task 13 is the corresponding jagged-utility experiment at seed 0. To inspect every task:

```sh
python triton_run_main_candidate_matched_task.py --task-csv triton_main_1d_official_seed0-9_tasks.csv --list-tasks
```

The default sampling settings are 1,000 tuning steps and 1,000 retained draws per chain, with four chains and target acceptance 0.95. `--cores 1` runs those chains sequentially; increase it to use parallel chains. Each task fits the full dataset and the trajectory prefixes `8, 16, 32, 64, 128, 256, 500`. These are research runs and can take substantial time.

Use a fresh result directory for a new experiment. Repeating the same task in the same directory can overwrite its results and prefix caches.

### Run the complete comparison

The published CSV already contains the full task grid. To regenerate it from the included task generator:

```sh
python triton_make_main_1d_official_tasks.py --seed-ids 0-9 --out triton_main_1d_official_seed0-9_tasks.csv
```

Run task IDs 1 through 160 with the same result root. For example, this PowerShell loop runs the entire suite sequentially and stops on a failed task:

```powershell
1..160 | ForEach-Object {
    python triton_run_main_candidate_matched_task.py --task-csv triton_main_1d_official_seed0-9_tasks.csv --task-id $_ --run-root results/main_1d --cores 1
    if ($LASTEXITCODE -ne 0) { throw "Task $_ failed." }
}
```

For cluster execution, submit one task ID per job using your cluster's environment and resource configuration.

### Aggregate and inspect results

After completing the required tasks, aggregate metrics across seeds:

```sh
python run_inference.py --aggregate-only --run-root results/main_1d --utility both
```

To recompute supported metrics and figures from saved compact posterior archives, then aggregate:

```sh
python run_inference.py --postprocess-only --run-root results/main_1d --utility both
```

Outputs include:

- Per-task configuration, summary, runtime, and artifact-completeness JSON files.
- Posterior utility summaries and trajectory-prefix metric tables under `tables/`.
- Compact raw posterior draws and sampling diagnostics under `posterior_archive/`.
- Static and interactive figures under `plots/`.
- Dataset-level tables aggregating metrics across seeds, plus `combined_run_summary.csv` at the result root.

Compact posterior archives are saved by default. Add `--save-full-idata` to an inference task to save full NetCDF inference data as well. HTML plots load Plotly from a CDN, so viewing them requires an internet connection. Review sampling diagnostics alongside the scientific metrics; successful execution alone does not establish convergence.

### Regenerate the synthetic inputs

The bundled inputs are sufficient for the official comparison. To regenerate the same simulation settings into a separate directory:

```sh
python generate_main_1d_candidates.py --out-dir results/regenerated_main_1d --utilities smooth,jagged --balanced-configs 0.97:10 --balanced-reps 5 --clipped-configs= --seed 0 --no-plots
```

The JSONL files are written under `results/regenerated_main_1d/data/`, together with a manifest and diagnostic tables in the output root. The empty `--clipped-configs=` argument disables the generator's additional peak-clipped candidates. Keep all of the explicit settings above: the generator's defaults describe a broader candidate sweep.

## 2D extension

The 2D editor varies headline and background grayscale, with state `(h, b)` on a `K x K` grid. The standard setup uses `K = 10` and a smooth contrast-interaction utility. The shared inference code supports the endpoint model, BIRL, and both preference baselines.

For a small functional example, first generate a reduced dataset on a `4 x 4` grid:

```sh
python run_experiments_2d.py --run-root outputs_2d/example --k 4 --utility smooth_contrast_interaction --dry-run --no-plots
```

In this generator, `--dry-run` **writes a small dataset** using reduced simulation budgets. This differs from the 1D task runner's validation-only `--dry-run`.

Then run a short endpoint-GP inference check:

```sh
python run_inference_2d.py --dataset outputs_2d/example/data/steps_UserModel4Boltzmann2D_gamma0.95_alpha5_train.jsonl --run-root outputs_2d/inference_example --grid-k 4 --variant-set m4-gp --seed-ids 0 --smoke
```

`--smoke` uses eight trajectories, five tuning steps, five retained draws, and one chain. It checks execution only; its estimates and convergence diagnostics are not suitable for scientific conclusions. For larger experiments, match the generator's `--k` and inference's `--grid-k`, use a fresh output root, and remove the reduced-budget flags.

## Code map

| Path | Role |
| --- | --- |
| `src/model.py` | Bayesian model and baseline definitions |
| `src/preprocess.py`, `src/state_space.py` | Trajectory preprocessing and 1D/2D state/action representations |
| `src/pipeline.py`, `src/sampling.py` | Inference orchestration and MCMC |
| `src/prefix_fit.py`, `src/prefix_analysis.py` | Learning curves from cumulative trajectory prefixes |
| `src/metrics.py`, `src/seed_aggregation.py` | Utility/recommendation metrics and aggregation across seeds |
| `src/posterior_archive.py`, `src/postprocess.py` | Posterior storage and postprocessing |
| `src/plots.py`, `src/plot_style.py` | Result figures and plotting conventions |
| `poster_env.py`, `user_models.py`, `utility.py` | 1D environment, simulated users, and ground-truth utilities |
| `simulate.py`, `sweep_runner.py`, `io_utils.py`, `plotting.py` | 1D simulation, output, and behavior diagnostics |
| `poster2d/` | 2D environment, utilities, simulated users, and smoke checks |
| `generate_main_1d_candidates.py`, `dataset_export_diagnostics.py` | Synthetic 1D inputs and export diagnostics |
| `triton_make_main_1d_official_tasks.py`, `triton_run_main_candidate_matched_task.py` | Official 1D task definition and execution |
| `run_inference.py` | General 1D batch runner, aggregation, and postprocessing |
| `run_experiments_2d.py`, `run_inference_2d.py` | 2D simulation and inference entry points |
| `tests/` | Unit tests for preprocessing, metrics, posterior archives, and inferred alpha |
