"""Block B of models_plan.md §25.4: tabular foundation models against the frozen T1.

TabPFN-3.5 and Kumo Tabular learn in context from the same 50,000 fit rows,
described by the first 100 components of the M6 PCA plus the product. They
are scored on the calibration rows without shared text that M5B used, and
each one wins only if its Macro-F1 beats the frozen T1 (BGE + product) by at
least 5% with a 95% weekly bootstrap interval above 0 (§25.2).

Weights are read from local files: the SLURM nodes have no internet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.bge_sample import ID_COLUMN
from src.models.representation_comparison import (
    attach_embedding_rows,
    bootstrap_metric,
    load_embeddings,
    load_frames,
    week_resamples,
    weighted_macro_f1,
)
from src.models.semantic_space import fixed_sample_positions, load_dvc_hash
from src.models.tfidf_models import eligible_mask

CANDIDATES = ("tabpfn", "kumo")
REFERENCE = "bge_product"
DIFFERENCE_QUANTILES = {"p025": 0.025, "p50": 0.50, "p975": 0.975}
SMOKE = {"context_rows": 2_000, "query_rows": 2_000, "estimators": 1, "draws": 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run both models on a few rows and save nothing.",
    )
    return parser.parse_args()


def check_sources(config: dict[str, Any], settings: dict[str, Any]) -> None:
    pointers = {
        config["paths"]["bge_full_artifacts_dvc"]: settings["bge_full_dvc_hash"],
        config["paths"]["semantic_artifacts_dvc"]: settings["semantic_dvc_hash"],
        settings["reference_dvc"]: settings["reference_dvc_hash"],
    }
    for path, frozen in pointers.items():
        if load_dvc_hash(PROJECT_ROOT / path) != frozen:
            raise ValueError(f"Block B source {path} does not match its frozen hash.")


def load_rows(
    config: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Context rows, calibration rows and the frozen T1 outputs, aligned."""
    frames = load_frames(config, smoke=False)
    fit = frames["fit"].loc[eligible_mask(frames["fit"], "T1", "complete")]
    positions = fixed_sample_positions(len(fit), settings["context_rows"], seed)
    context = fit.iloc[positions].reset_index(drop=True)
    calibration = frames["calibration"]
    query = calibration.loc[eligible_mask(calibration, "T1", "complete")]
    query = query.reset_index(drop=True)
    reference = pd.read_parquet(PROJECT_ROOT / settings["reference_predictions"])
    if not np.array_equal(
        query[ID_COLUMN].astype(str).to_numpy(),
        reference[ID_COLUMN].astype(str).to_numpy(),
    ):
        raise ValueError("Block B calibration rows differ from the M5B rows.")
    selected = {"fit": context, "calibration": query}
    manifest = PROJECT_ROOT / config["paths"]["bge_full_artifacts"] / "manifest.parquet"
    attach_embedding_rows(selected, manifest)
    return context, query, reference


def features(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    components: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """First PCA components of each row's BGE embedding, and its product code."""
    embeddings = load_embeddings(
        PROJECT_ROOT / config["paths"]["bge_full_artifacts"], frames
    )
    semantic_dir = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    pca = joblib.load(semantic_dir / "pca.joblib")
    # M4's encoder knows every fit product; the context sample may miss some.
    tfidf_dir = PROJECT_ROOT / config["paths"]["tfidf_models"]
    encoder = joblib.load(tfidf_dir / "product_encoder.joblib")
    products = [str(product) for product in encoder.categories_[0]]
    numerical = {}
    codes = {}
    for split, values in embeddings.items():
        numerical[split] = pca.transform(values)[:, :components].astype(np.float32)
        product = pd.Categorical(
            frames[split][CANONICAL_PRODUCT_COLUMN].astype(str), categories=products
        )
        if (product.codes < 0).any():
            raise ValueError(f"Block B found a {split} product unknown to M4.")
        codes[split] = product.codes.astype(np.int64)
    return numerical, codes, products


def check_reference(metrics: dict[str, float], config: dict[str, Any]) -> None:
    """The frozen T1 must reproduce its published M5B calibration Macro-F1."""
    path = PROJECT_ROOT / config["paths"]["representation_report"]
    published = json.loads(path.read_text(encoding="utf-8"))["results"]["T1"]
    expected = published["metrics"][REFERENCE]["macro_f1"]
    if not np.isclose(metrics["macro_f1"], expected, rtol=0, atol=1e-9):
        raise ValueError(
            f"Block B does not reproduce the frozen T1: {metrics['macro_f1']} vs "
            f"{expected}."
        )


def top_three(
    probability: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Most likely class and the three most likely classes of each row."""
    order = np.argsort(-probability, axis=1)[:, :3]
    return classes[order[:, 0]], classes[order]


def run_tabpfn(
    numerical: dict[str, np.ndarray],
    codes: dict[str, np.ndarray],
    labels: np.ndarray,
    settings: dict[str, Any],
    estimators: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from tabpfn import TabPFNClassifier

    def table(split: str) -> np.ndarray:
        return np.column_stack([numerical[split], codes[split]]).astype(np.float32)

    model = TabPFNClassifier(
        model_path=str(Path(settings["tabpfn_model_path"]).expanduser()),
        device="cuda",
        n_estimators=estimators,
        categorical_features_indices=[numerical["fit"].shape[1]],
        random_state=seed,
    )
    model.fit(table("fit"), labels)
    query = table("calibration")
    batch = settings["query_batch_rows"]
    probability = np.vstack(
        [
            model.predict_proba(query[start : start + batch])
            for start in range(0, len(query), batch)
        ]
    )
    return probability, np.asarray(model.classes_)


def run_kumo(
    numerical: dict[str, np.ndarray],
    codes: dict[str, np.ndarray],
    products: list[str],
    labels: np.ndarray,
    settings: dict[str, Any],
    estimators: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from sdm import CategoricalTensor, StringTensor, TableTensor
    from sdm.models import KumoTabular

    names = [f"pc{index}" for index in range(numerical["fit"].shape[1])]

    # Tables are built on the CPU and moved whole: every block must share a device.
    def table(split: str, rows: slice = slice(None)) -> TableTensor:
        return TableTensor(
            columns={"numerical": names, "categorical": ["product"]},
            numerical=torch.from_numpy(numerical[split][rows]),
            categorical=CategoricalTensor(
                code=torch.from_numpy(codes[split][rows, None]),
                categories=(StringTensor.from_list(products),),
            ),
        ).to("cuda")

    classes = np.unique(labels)
    target = TableTensor(
        columns={"categorical": ["T1"]},
        categorical=CategoricalTensor(
            code=torch.from_numpy(np.searchsorted(classes, labels)[:, None]),
            categories=(StringTensor.from_list(classes.tolist()),),
        ),
    ).to("cuda")
    model = KumoTabular(
        task="classification", size=settings["kumo_size"], device="cuda"
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    context = table("fit")
    batch = settings["query_batch_rows"]
    parts = []
    with torch.no_grad():
        for start in range(0, len(numerical["calibration"]), batch):
            output = model(
                context,
                target,
                table("calibration", slice(start, start + batch)),
                num_estimators=estimators,
                generator=generator,
            ).numerical
            # One column per context class, in the order of the target categories.
            if output.dim() == 3:
                output = output.mean(dim=0)
            parts.append(output.float().cpu().numpy())
    probability = np.vstack(parts)
    if probability.shape[1] != len(classes):
        raise ValueError("Kumo Tabular returned an unexpected number of classes.")
    return probability, classes


def compare(
    outputs: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
    weeks: np.ndarray,
    classes: np.ndarray,
    minimum_gain: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Macro-F1 and top-3 of each model, and the §25.2 rule against the reference."""
    truth_codes = pd.Categorical(truth, categories=classes).codes

    def macro_f1(name: str) -> Any:
        prediction = outputs[name]["prediction"]
        codes = pd.Categorical(prediction, categories=classes).codes
        return lambda weights: weighted_macro_f1(
            truth_codes, codes, weights, len(classes)
        )

    counts, positions = week_resamples(weeks, draws, seed)
    ones = np.ones(len(truth))
    metrics = {
        name: {
            "macro_f1": macro_f1(name)(ones),
            "top_three": float(np.mean(values["top_three"])),
        }
        for name, values in outputs.items()
    }
    reference_draws = bootstrap_metric(macro_f1(REFERENCE), counts, positions)
    comparisons = {}
    for name in outputs:
        if name == REFERENCE:
            continue
        candidate_draws = bootstrap_metric(macro_f1(name), counts, positions)
        difference = candidate_draws - reference_draws
        quantiles = np.quantile(difference, list(DIFFERENCE_QUANTILES.values()))
        interval = {
            label: float(value) for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
        }
        gain = metrics[name]["macro_f1"] / metrics[REFERENCE]["macro_f1"] - 1
        comparisons[name] = {
            "relative_gain": float(gain),
            "difference": {
                "estimate": metrics[name]["macro_f1"] - metrics[REFERENCE]["macro_f1"],
                **interval,
            },
            "wins": bool(gain >= minimum_gain and interval["p025"] > 0),
        }
    weeks_used = int(counts.shape[1])
    return {"metrics": metrics, "comparisons": comparisons, "weeks": weeks_used}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["foundation_t1"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    check_sources(config, settings)
    report_path = PROJECT_ROOT / settings["report"]
    artifact_dir = PROJECT_ROOT / settings["artifacts"]
    if not args.smoke and (report_path.exists() or artifact_dir.exists()):
        raise FileExistsError("Block B never overwrites its results.")

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
    numerical, codes, products = features(frames, config, settings["pca_components"])
    labels = context["T1"].astype(str).to_numpy()
    print(f"Inputs: {len(context)} context, {len(query)} query rows.", flush=True)

    outputs = {
        REFERENCE: {
            "prediction": reference[f"{REFERENCE}_prediction"].astype(str).to_numpy(),
            "top_three": reference[f"{REFERENCE}_top_three"].to_numpy(dtype=bool),
        }
    }
    truth = reference["truth"].astype(str).to_numpy()
    seconds = {}
    for name in CANDIDATES:
        candidate_started = time.perf_counter()
        if name == "tabpfn":
            probability, classes = run_tabpfn(
                numerical, codes, labels, settings, estimators, seed
            )
        else:
            probability, classes = run_kumo(
                numerical, codes, products, labels, settings, estimators, seed
            )
        prediction, three = top_three(probability, classes)
        outputs[name] = {
            "prediction": prediction.astype(str),
            "top_three": (three.astype(str) == truth[:, None]).any(axis=1),
        }
        seconds[name] = time.perf_counter() - candidate_started
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
        print("Block B smoke finished; nothing was saved.")
        return
    check_reference(result["metrics"][REFERENCE], config)

    report = {
        "stage": "M5F",
        "plan": "models_plan.md §25.4",
        "git_commit": git_commit(),
        "seed": seed,
        "validation_used": False,
        "reference": REFERENCE,
        "context_rows": len(context),
        "context_classes": int(len(np.unique(labels))),
        "query_rows": len(query),
        "pca_components": settings["pca_components"],
        "estimators": estimators,
        "bootstrap_draws": draws,
        **result,
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
        run_name="m5f-foundation-t1",
        stage="M5F",
        target="T1",
        split="calibration",
        view="no_shared_text",
        features=["BGE PCA 100", "product"],
        parameters={
            "plan": report["plan"],
            "context_rows": report["context_rows"],
            "estimators": report["estimators"],
            "tabpfn_weights": Path(settings["tabpfn_model_path"]).name,
            "kumo_size": settings["kumo_size"],
            "winners": json.dumps(
                [name for name, value in report["comparisons"].items() if value["wins"]]
            ),
        },
        metrics=metrics,
        artifacts=[settings["report"]],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m5f"
    save_run_record(record, run_root / "foundation_t1" / "run.json")


if __name__ == "__main__":
    main()
