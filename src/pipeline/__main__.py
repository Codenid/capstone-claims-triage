"""Command line of the pipeline (models_plan.md §29).

python -m src.pipeline fit [--run-preparation] [--report ruta.json]
python -m src.pipeline score --input tabla.parquet --output carpeta/ [--device cuda]
python -m src.pipeline sample --start 2024-11-04 --end 2024-11-17 --output tabla.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.experiment import PROJECT_ROOT, load_experiment_config
from src.pipeline.contract import ID_COLUMN, read_table
from src.pipeline.fit import fit_report, print_fit_report, write_fit_report
from src.pipeline.score import score_table

SAMPLE_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Company response to consumer",
    "Timely response?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit", help="Check the frozen chain and write its manifest."
    )
    fit.add_argument(
        "--run-preparation", action="store_true", help="Run the DVC stages P1-P6."
    )
    fit.add_argument("--report", type=Path, default=Path("reports/pipeline/fit.json"))

    score = commands.add_parser(
        "score", help="Score a new table with the frozen components."
    )
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument(
        "--device", default="cpu", help="cpu or cuda, for BGE and TabPFN"
    )
    score.add_argument("--neighbors", type=int, default=5)
    score.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Use the linear T1 even with weights.",
    )

    sample = commands.add_parser(
        "sample", help="Cut a date range of the prepared table as a demo input."
    )
    sample.add_argument("--start", required=True)
    sample.add_argument("--end", required=True, help="exclusive")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def sample_input(start: str, end: str, output: Path, max_rows: int) -> int:
    """Rows of the prepared table in [start, end) with the public CFPB columns only."""
    config = load_experiment_config()
    table = pq.read_table(
        PROJECT_ROOT / config["paths"]["input_data"],
        columns=SAMPLE_COLUMNS,
        filters=[
            ("Date received", ">=", pd.Timestamp(start).date()),
            ("Date received", "<", pd.Timestamp(end).date()),
        ],
    )
    frame = table.to_pandas()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    if max_rows:
        frame = frame.head(max_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return len(frame)


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        config = load_experiment_config()
        report = fit_report(config, args.run_preparation)
        print_fit_report(report)
        write_fit_report(report, PROJECT_ROOT / args.report)
        return
    if args.command == "sample":
        rows = sample_input(args.start, args.end, args.output, args.max_rows)
        print(f"{rows} filas -> {args.output}")
        return
    config = load_experiment_config()
    frame = read_table(args.input)
    report = score_table(
        frame,
        config,
        args.output,
        device=args.device,
        neighbors=args.neighbors,
        skip_tabpfn=args.skip_tabpfn,
    )
    print(
        f"Filas: {report['rows']} · semanas: {report['weeks']} · "
        f"días: {report['days']} · T1: {report['t1_model']}"
    )
    print(f"Alertas: {report['alerts']}")
    signals = report["signals"]
    print(
        f"Señales: reentrenar={signals['retrain_recommended']} {signals['warnings']}"
    )
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
