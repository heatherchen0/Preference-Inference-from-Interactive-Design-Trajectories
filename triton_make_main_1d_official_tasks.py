from __future__ import annotations

import argparse
import csv
from pathlib import Path


PREFIX_K_TRAJ_LIST = (8, 16, 32, 64, 128, 256, 500)

DATASETS = [
    {
        "dataset_index": 1,
        "dataset_label": "smooth-gamma0.97-alpha10-reps5-seed0",
        "utility_kind": "smooth",
        "train_file": "data/main_1d/steps_UserModel4Boltzmann_gamma0.97_alpha10_reps5_uniform_train.jsonl",
        "gamma_fixed": 0.97,
        "alpha_fixed": 10.0,
    },
    {
        "dataset_index": 2,
        "dataset_label": "jagged-gamma0.97-alpha10-reps5-seed0",
        "utility_kind": "jagged",
        "train_file": "data/main_1d/steps_Jagged_UserModel4Boltzmann_gamma0.97_alpha10_reps5_uniform_train.jsonl",
        "gamma_fixed": 0.97,
        "alpha_fixed": 10.0,
    },
]

VARIANTS = [
    {
        "variant_index": 1,
        "model_name": "pbo_baseline_start_end_preference",
        "likelihood": "start_end_logistic",
        "prior": "gp_rbf",
    },
    {
        "variant_index": 2,
        "model_name": "pbo_baseline_start_end_preference",
        "likelihood": "start_end_logistic",
        "prior": "iid",
    },
    {
        "variant_index": 3,
        "model_name": "pbo_baseline_transition_preference",
        "likelihood": "transition_logistic",
        "prior": "gp_rbf",
    },
    {
        "variant_index": 4,
        "model_name": "pbo_baseline_transition_preference",
        "likelihood": "transition_logistic",
        "prior": "iid",
    },
    {
        "variant_index": 5,
        "model_name": "model4_endpoint_discounted_boltzmann",
        "likelihood": "boltzmann_qstar",
        "prior": "gp_rbf",
    },
    {
        "variant_index": 6,
        "model_name": "model4_endpoint_discounted_boltzmann",
        "likelihood": "boltzmann_qstar",
        "prior": "iid",
    },
    {
        "variant_index": 7,
        "model_name": "birl_baseline_discounted_boltzmann",
        "likelihood": "boltzmann_qstar",
        "prior": "gp_rbf",
    },
    {
        "variant_index": 8,
        "model_name": "birl_baseline_discounted_boltzmann",
        "likelihood": "boltzmann_qstar",
        "prior": "iid",
    },
]


def _parse_seed_ids(text: str) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            vals = range(int(lo_s), int(hi_s) + 1)
        else:
            vals = (int(part),)
        for value in vals:
            if value < 0:
                raise ValueError(f"Seed ids must be non-negative, got {value}.")
            if value not in seen:
                seen.add(value)
                seeds.append(value)
    if not seeds:
        raise ValueError("--seed-ids must contain at least one seed.")
    return seeds


def _model_code(model_name: str) -> str:
    name = str(model_name).lower()
    if name == "model4_endpoint_discounted_boltzmann":
        return "m4"
    if name == "birl_baseline_discounted_boltzmann":
        return "birl"
    if name in {
        "pbo_baseline",
        "pbo_baseline_start_end_preference",
        "pbo_baseline_transition_preference",
    }:
        return "pbo"
    return name


def _likelihood_code(likelihood: str) -> str:
    likelihood = str(likelihood).lower()
    if likelihood == "boltzmann_qstar":
        return "boltz"
    return likelihood


def _variant_label(*, prior: str, model_name: str, likelihood: str) -> str:
    return f"{_model_code(model_name)}-{_likelihood_code(likelihood)}-{prior}"


def _variant_label_from_row(row: dict) -> str:
    return _variant_label(
        prior=str(row["prior"]),
        model_name=str(row["model_name"]),
        likelihood=str(row["likelihood"]),
    )


def _build_rows(seed_ids: list[int]) -> list[dict]:
    rows: list[dict] = []
    task_id = 1
    prefix_list = ";".join(str(k) for k in PREFIX_K_TRAJ_LIST)
    for seed_id in seed_ids:
        for dataset in DATASETS:
            for variant in VARIANTS:
                rows.append(
                    {
                        "task_id": task_id,
                        "task_family": "main_1d_official_matched",
                        "dataset_index": dataset["dataset_index"],
                        "dataset_label": dataset["dataset_label"],
                        "utility_kind": dataset["utility_kind"],
                        "train_file": dataset["train_file"],
                        "gamma_fixed": f"{float(dataset['gamma_fixed']):g}",
                        "alpha_fixed": f"{float(dataset['alpha_fixed']):g}",
                        "seed_id": int(seed_id),
                        "variant_index": variant["variant_index"],
                        "variant_label": _variant_label_from_row(variant),
                        "model_name": variant["model_name"],
                        "likelihood": variant["likelihood"],
                        "prior": variant["prior"],
                        "prefix_schedule": "explicit",
                        "prefix_k_traj_list": prefix_list,
                    }
                )
                task_id += 1
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Triton task CSV for the official 1D main datasets."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton_main_1d_official_seed0-9_tasks.csv"),
    )
    parser.add_argument(
        "--seed-ids",
        default="0-9",
        help="Seed ids in comma/range syntax, e.g. '0-9' or '0,1,2'.",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Write the CSV even if an official dataset file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_ids = _parse_seed_ids(str(args.seed_ids))

    missing = [
        str(row["train_file"])
        for row in DATASETS
        if not Path(str(row["train_file"])).exists()
    ]
    if missing and not bool(args.allow_missing_data):
        raise FileNotFoundError(
            "Missing official dataset files:\n" + "\n".join(missing)
        )

    rows = _build_rows(seed_ids)
    fieldnames = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{args.out}: {len(rows)} tasks")
    print(f"datasets={len(DATASETS)} variants={len(VARIANTS)} seeds={len(seed_ids)}")
    print(f"prefix_k_traj_list={','.join(str(k) for k in PREFIX_K_TRAJ_LIST)}")


if __name__ == "__main__":
    main()
