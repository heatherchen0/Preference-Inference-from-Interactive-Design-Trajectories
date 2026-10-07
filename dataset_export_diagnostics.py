from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sort_key(row: dict):
    return (row["traj_id"], int(row["t"]))


def summarize_file(path: str | Path) -> dict:
    path = Path(path)
    rows = _read_jsonl(path)
    rows = sorted(rows, key=_sort_key)

    trajs: list[list[dict]] = []
    current_tid = None
    current: list[dict] = []

    for row in rows:
        tid = row["traj_id"]
        if current_tid is None or tid != current_tid:
            if current:
                trajs.append(current)
            current_tid = tid
            current = []
        current.append(row)

    if current:
        trajs.append(current)

    n_traj = len(trajs)
    n_obs = len(rows)
    n_immediate_export = 0
    n_start_eq_export = 0
    n_nontrivial_pbo_duels = 0
    lengths = []

    for traj in trajs:
        first = traj[0]
        last = traj[-1]
        start = int(first.get("start_poster", first["poster"]))
        end = int(last["poster"])
        length = len(traj)
        lengths.append(length)

        first_action = str(first.get("action", "")).strip().lower()
        if length == 1 and first_action in ("export", "stay"):
            n_immediate_export += 1

        if start == end:
            n_start_eq_export += 1
        else:
            n_nontrivial_pbo_duels += 1

    mean_len = sum(lengths) / n_traj if n_traj else float("nan")

    return {
        "file": str(path),
        "n_traj": int(n_traj),
        "n_obs": int(n_obs),
        "mean_rows_per_traj": float(mean_len),
        "n_immediate_export": int(n_immediate_export),
        "frac_immediate_export": float(n_immediate_export / n_traj) if n_traj else float("nan"),
        "n_start_eq_export": int(n_start_eq_export),
        "frac_start_eq_export": float(n_start_eq_export / n_traj) if n_traj else float("nan"),
        "n_nontrivial_pbo_duels": int(n_nontrivial_pbo_duels),
        "frac_nontrivial_pbo_duels": float(n_nontrivial_pbo_duels / n_traj) if n_traj else float("nan"),
    }


def _format_float(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def print_table(rows: list[dict]) -> None:
    headers = [
        "file",
        "n_traj",
        "n_obs",
        "mean_len",
        "imm_export",
        "imm_frac",
        "start=end",
        "start=end_frac",
        "pbo_duels",
        "pbo_duel_frac",
    ]

    table = []
    for r in rows:
        table.append(
            [
                Path(r["file"]).name,
                str(r["n_traj"]),
                str(r["n_obs"]),
                _format_float(r["mean_rows_per_traj"]),
                str(r["n_immediate_export"]),
                _format_float(r["frac_immediate_export"]),
                str(r["n_start_eq_export"]),
                _format_float(r["frac_start_eq_export"]),
                str(r["n_nontrivial_pbo_duels"]),
                _format_float(r["frac_nontrivial_pbo_duels"]),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        for i in range(len(headers))
    ]

    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize immediate-export and start==export rates in trajectory JSONL datasets."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL file paths or glob patterns, e.g. steps_UserModel4Boltzmann*.jsonl",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a compact table.",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.paths:
        matches = sorted(Path(".").glob(item))
        if matches:
            files.extend(matches)
        else:
            files.append(Path(item))

    seen = set()
    unique_files = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            unique_files.append(f)

    rows = [summarize_file(f) for f in unique_files]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
