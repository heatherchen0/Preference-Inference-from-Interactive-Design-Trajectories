from __future__ import annotations

from pathlib import Path


MAIN_UTILITIES = ("smooth", "jagged")
ROBUSTNESS_DATASET_INDICES = (1, 2)
SEEDS = tuple(range(5))
INFER_ALPHA_VARIANT_INDEX = 1


def make_main_tasks(
    *,
    utilities=MAIN_UTILITIES,
    seeds=SEEDS,
    variant_index: int = INFER_ALPHA_VARIANT_INDEX,
) -> list[tuple[str, int, int]]:
    return [
        (str(utility), int(seed), int(variant_index))
        for utility in utilities
        for seed in seeds
    ]


def make_robustness_tasks(
    *,
    dataset_indices=ROBUSTNESS_DATASET_INDICES,
    seeds=SEEDS,
    variant_index: int = INFER_ALPHA_VARIANT_INDEX,
) -> list[tuple[int, int, int]]:
    return [
        (int(dataset_index), int(seed), int(variant_index))
        for dataset_index in dataset_indices
        for seed in seeds
    ]


def _write_rows(path: str | Path, header: str, rows: list[tuple]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"{path}: {len(rows)} tasks")


def main() -> None:
    _write_rows(
        "triton_infer_alpha_1d_tasks.csv",
        "utility,seed_id,variant_index",
        make_main_tasks(),
    )
    _write_rows(
        "triton_infer_alpha_robustness_tasks.csv",
        "dataset_index,seed_id,variant_index",
        make_robustness_tasks(),
    )


if __name__ == "__main__":
    main()
