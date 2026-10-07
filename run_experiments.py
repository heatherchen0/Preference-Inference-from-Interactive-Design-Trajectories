import numpy as np
from datetime import datetime
from poster_env import Problem1D, G_MIN, G_MAX
from utility import utility, jagged_utility
from user_models import UserModel2Discounted, UserModel4_EndpointDiscountedBoltzmann
from simulate import simulate_trajectories_with_random_starts, trajectory_summary, summarize_exports
from io_utils import save_jsonl, save_csv
from sweep_runner import run_systematic_sweep


def run_systematic_experiment(
    problem,
    user,
    utility_fn,
    label,
    starts,
    n_reps_per_start=100,
    max_steps=10_000,
    out_dir=".",
    shuffle_trajectories=True,
    rng=None,
):
    sweep = run_systematic_sweep(
        problem=problem,
        user=user,
        utility_fn=utility_fn,
        starts=starts,
        n_reps_per_start=n_reps_per_start,
        max_steps=max_steps,
        shuffle_trajectories=shuffle_trajectories,
        rng=rng,
    )

    step_records = []
    traj_id = 0
    for item in sweep["trajectories"]:
        start_poster = int(item["start_poster"])
        rep = int(item["rep"])
        traj = item["traj"]

        for r in traj:
            step_records.append(
                {
                    "traj_id": int(traj_id),
                    "t": int(r["t"]),
                    "poster": int(r["poster"]),
                    "action": r["action"],
                    "start_poster": start_poster,
                    "rep": rep,
                    "model": label,
                }
            )
        traj_id += 1

    # Per-trajectory summaries
    per_traj_rows = []
    for r in sweep["per_traj_rows"]:
        rr = dict(r)
        rr["model"] = label
        per_traj_rows.append(rr)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = f"{out_dir}/steps_{label}_{ts}.jsonl"
    save_jsonl(jsonl_path, step_records)

    print(f"\n=== {label} (balanced starts, shuffled trajectories={shuffle_trajectories}) ===")
    print("Saved:")
    print(" ", jsonl_path)

    return per_traj_rows, step_records


if __name__ == "__main__":

# model 2
    # problem = Problem1D(g_min=G_MIN, g_max=G_MAX, g0=1)

    # starts = [1, 10, 20, 35, 45, 60, 85, 95]
    # gammas = [0.9, 0.95, 0.995]

    # n_reps_per_start = 1
    # max_steps = 10_000
    # out_dir = "."
    # seed0 = 0

    # merged_rows = []

    # for gamma in gammas:
    #     label = f"UserModel2Discounted_gamma{gamma}"

    #     user2 = UserModel2Discounted(
    #         utility_fn=utility,
    #         g_min=G_MIN,
    #         g_max=G_MAX,
    #         gamma=float(gamma),
    #     )

    #     per_traj_rows, _step_records = run_systematic_experiment(
    #         problem=problem,
    #         user=user2,
    #         utility_fn=utility,
    #         label=label,
    #         starts=starts,
    #         n_reps_per_start=n_reps_per_start,
    #         max_steps=max_steps,
    #         out_dir=out_dir,
    #     )

#model 4
    problem = Problem1D(g_min=G_MIN, g_max=G_MAX, g0=1)
    starts = list(range(G_MIN, G_MAX + 1))

    gammas = [0.95,0.99]
    alphas = [10,15]

    n_reps_train = 3 #16
    n_reps_test = 3
    max_steps = 10_000
    out_dir = "."
    seed0 = 1

    for gamma in gammas:
        for alpha in alphas:
            rng_sweep_train = np.random.default_rng(seed0 + 1000)
            rng_sweep_test  = np.random.default_rng(seed0 + 2000)

            user_train = UserModel4_EndpointDiscountedBoltzmann(
                utility_fn=utility,
                g_min=G_MIN,
                g_max=G_MAX,
                gamma=float(gamma),
                alpha=float(alpha),
                rng=np.random.default_rng(seed0 + 3000),
            )

            # user_test = UserModel4_EndpointDiscountedBoltzmann(
            #     utility_fn=jagged_utility,
            #     g_min=G_MIN,
            #     g_max=G_MAX,
            #     gamma=float(gamma),
            #     alpha=float(alpha),
            #     rng=np.random.default_rng(seed0 + 4000),
            # )

            label_train = f"UserModel4Boltzmann_gamma{gamma}_alpha{alpha}_train"
            # label_test  = f"UserModel4Boltzmann_gamma{gamma}_alpha{alpha}_test"

            run_systematic_experiment(
                problem=problem,
                user=user_train,
                utility_fn=utility,
                label=label_train,
                starts=starts,
                n_reps_per_start=n_reps_train,
                max_steps=max_steps,
                out_dir=out_dir,
                shuffle_trajectories=True,
                rng=rng_sweep_train,
            )

    #         # run_systematic_experiment(
    #         #     problem=problem,
    #         #     user=user_test,
    #         #     utility_fn=utility,
    #         #     label=label_test,
    #         #     starts=starts,
    #         #     n_reps_per_start=n_reps_test,
    #         #     max_steps=max_steps,
    #         #     out_dir=out_dir,
    #         #     shuffle_trajectories=True,
    #         #     rng=rng_sweep_test,
    #         # )

# model 2    
#     problem = Problem1D(g_min=G_MIN, g_max=G_MAX, g0=1)
#     starts = list(range(G_MIN, G_MAX + 1))

#     gammas = [0.95]

#     n_reps_train = 1 #16
#     n_reps_test = 3
#     max_steps = 10_000
#     out_dir = "."
#     seed0 = 0

#     for gamma in gammas:
#         rng_sweep_train = np.random.default_rng(seed0 + 1000)
#         #rng_sweep_test  = np.random.default_rng(seed0 + 2000)

#         user_train = UserModel2Discounted(
#             utility_fn=utility,
#             g_min=G_MIN,
#             g_max=G_MAX,
#             gamma=float(gamma),
#         )

    #     # user_test = UserModel4_EndpointDiscountedBoltzmann(
    #     #     utility_fn=utility,
    #     #     g_min=G_MIN,
    #     #     g_max=G_MAX,
    #     #     gamma=float(gamma),
    #     #     alpha=float(alpha),
    #     #     rng=np.random.default_rng(seed0 + 4000),
    #     # )

    #     label_train = f"UserModel2_gamma{gamma}_fixalpha10_train"
    #     #label_test  = f"UserModel4Boltzmann_gamma{gamma}_alpha{alpha}_test"

    #     run_systematic_experiment(
    #         problem=problem,
    #         user=user_train,
    #         utility_fn=utility,
    #         label=label_train,
    #         starts=starts,
    #         n_reps_per_start=n_reps_train,
    #         max_steps=max_steps,
    #         out_dir=out_dir,
    #         shuffle_trajectories=True,
    #         rng=rng_sweep_train,
    #     )