"""Input ablation of TabPFN-3.5 on T1, models_plan.md §26.

The block B winner (§25.4) saw the first 100 components of the M6 PCA plus the
product. Each variant here changes only that numerical input: all 256 frozen
PCA components, or the 1,024 BGE dimensions without the PCA. Context rows,
estimators, seed and query rows stay as in block B, and TabPFN-100's saved
calibration predictions are the reference. A variant replaces TabPFN-100 only
by the §25.2 rule: at least 5% more Macro-F1 with the 95% weekly bootstrap
interval above 0.

Nothing here touches the frozen T1 of M5B; the ablation is TabPFN against
TabPFN.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.bge_sample import ID_COLUMN
from src.models.foundation_t1 import (
    SMOKE,
    check_sources,
    compare,
    features,
    load_rows,
    run_tabpfn,
    top_three,
)
from src.models.representation_comparison import read_json
from src.models.semantic_space import fixed_sample_positions

REFERENCE = "tabpfn"
STAGE = "M5F"
PLAN = "models_plan.md §26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run every variant on a few rows and save nothing.",
    )
    return parser.parse_args()


def load_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Block B settings with the ablation's own reference and outputs on top."""
    return config["foundation_t1"] | config["foundation_t1_ablation"]


def check_reference(metrics: dict[str, float], settings: dict[str, Any]) -> None:
    """The saved TabPFN-100 predictions must reproduce their published Macro-F1."""
    published = read_json(PROJECT_ROOT / settings["report"])
    expected = published["metrics"][REFERENCE]["macro_f1"]
    if not np.isclose(metrics["macro_f1"], expected, rtol=0, atol=1e-9):
        raise ValueError(
            f"The ablation does not reproduce TabPFN-100: {metrics['macro_f1']} "
            f"vs {expected}."
        )


def input_label(components: int | None) -> str:
    return "BGE 1024" if components is None else f"BGE PCA {components}"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_experiment_config()
    settings = load_settings(config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    check_sources(config, settings)
    report_path = PROJECT_ROOT / settings["ablation_report"]
    artifact_dir = PROJECT_ROOT / settings["ablation_artifacts"]
    if not args.smoke and (report_path.exists() or artifact_dir.exists()):
        raise FileExistsError("The ablation never overwrites its results.")

    context, query, reference = load_rows(config, settings, seed)
    estimators = settings["estimators"]
    draws = settings["bootstrap_draws"]
    if args.smoke:
        context = context.iloc[: SMOKE["context_rows"]]
        keep = fixed_sample_positions(len(query), SMOKE["query_rows"], seed)
        query = query.iloc[keep].reset_index(drop=True)
        reference = reference.iloc[keep].reset_index(drop=True)
        estimators = SMOKE["estimators"]
        draws = SMOKE["draws"]
    frames = {"fit": context, "calibration": query}
    labels = context["T1"].astype(str).to_numpy()
    truth = reference["truth"].astype(str).to_numpy()
    print(f"Inputs: {len(context)} context, {len(query)} query rows.", flush=True)

    outputs = {
        REFERENCE: {
            "prediction": reference[f"{REFERENCE}_prediction"].astype(str).to_numpy(),
            "top_three": reference[f"{REFERENCE}_top_three"].to_numpy(dtype=bool),
        }
    }
    seconds = {}
    for name, components in settings["inputs"].items():
        variant_started = time.perf_counter()
        numerical, codes, _ = features(frames, config, components)
        print(f"{name}: {numerical['fit'].shape[1]} numerical columns", flush=True)
        probability, classes = run_tabpfn(
            numerical, codes, labels, settings, estimators, seed
        )
        prediction, three = top_three(probability, classes)
        outputs[name] = {
            "prediction": prediction.astype(str),
            "top_three": (three.astype(str) == truth[:, None]).any(axis=1),
        }
        seconds[name] = time.perf_counter() - variant_started
        print(f"{name}: {seconds[name]:.0f}s", flush=True)

    all_classes = np.unique(
        np.concatenate([truth, *(values["prediction"] for values in outputs.values())])
    )
    result = compare(
        outputs,
        truth,
        pd.to_datetime(query["week"]).to_numpy(),
        all_classes,
        settings["minimum_relative_gain"],
        draws,
        seed,
        reference=REFERENCE,
    )
    for name, values in result["metrics"].items():
        print(
            f"{name}: Macro-F1={values['macro_f1']:.4f} "
            f"top-3={values['top_three']:.4f}"
        )
    for name, comparison in result["comparisons"].items():
        difference = comparison["difference"]
        print(
            f"{name}: gain={comparison['relative_gain']:+.1%} "
            f"95% [{difference['p025']:+.4f}, {difference['p975']:+.4f}] "
            f"wins={comparison['wins']}"
        )
    if args.smoke:
        print("Ablation smoke finished; nothing was saved.")
        return
    check_reference(result["metrics"][REFERENCE], settings)

    winners = [name for name, value in result["comparisons"].items() if value["wins"]]
    best = max(
        winners, key=lambda name: result["metrics"][name]["macro_f1"], default=None
    )
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "git_commit": git_commit(),
        "seed": seed,
        "validation_used": False,
        "reference": REFERENCE,
        "reference_pca_components": settings["pca_components"],
        "inputs": {
            name: {"pca_components": components, "label": input_label(components)}
            for name, components in settings["inputs"].items()
        },
        "context_rows": len(context),
        "context_classes": int(len(np.unique(labels))),
        "query_rows": len(query),
        "estimators": estimators,
        "bootstrap_draws": draws,
        **result,
        "winners": winners,
        "best_winner": best,
        "elapsed_seconds": {**seconds, "total": time.perf_counter() - started},
    }
    artifact_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            ID_COLUMN: query[ID_COLUMN].astype(str).to_numpy(),
            "truth": truth,
            **{
                f"{name}_{key}": values[key]
                for name, values in outputs.items()
                if name != REFERENCE
                for key in ("prediction", "top_three")
            },
        }
    ).to_parquet(artifact_dir / "calibration_t1.parquet", index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_offline_record(config, settings, report)
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


def save_offline_record(
    config: dict[str, Any],
    settings: dict[str, Any],
    report: dict[str, Any],
) -> None:
    metrics = {
        f"calibration_no_shared_text_{name}_{metric}": float(value)
        for name, values in report["metrics"].items()
        for metric, value in values.items()
    }
    for name, comparison in report["comparisons"].items():
        metrics[f"bootstrap_{name}_relative_gain"] = comparison["relative_gain"]
        for label, value in comparison["difference"].items():
            metrics[f"bootstrap_{name}_difference_{label}"] = value
    record = build_run_record(
        config=config,
        run_name="m5f-foundation-t1-ablation",
        stage=STAGE,
        target="T1",
        split="calibration",
        view="no_shared_text",
        features=[value["label"] for value in report["inputs"].values()] + ["product"],
        parameters={
            "plan": PLAN,
            "reference": f"tabpfn pca {report['reference_pca_components']}",
            "context_rows": report["context_rows"],
            "estimators": report["estimators"],
            "tabpfn_weights": os.path.basename(settings["tabpfn_model_path"]),
            "winners": json.dumps(report["winners"]),
            "best_winner": json.dumps(report["best_winner"]),
        },
        metrics=metrics,
        artifacts=[settings["ablation_report"]],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m5f"
    save_run_record(record, run_root / "foundation_t1_ablation" / "run.json")


if __name__ == "__main__":
    main()
