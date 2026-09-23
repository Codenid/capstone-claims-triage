"""Compare BGE and TF-IDF on the same temporal sample."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.build_targets import COMPLETE_COLUMNS
from src.data.normalize_text import HASH_COLUMN, NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.frequency_baselines import flat_mlflow_metrics
from src.models.tfidf_models import (
    DATE_COLUMN,
    NO_SHARED_COLUMN,
    TARGETS,
    build_features,
    train_models,
)

ID_COLUMN = "Complaint ID"


def load_source_rows(input_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    splits = config["evaluation"]["splits"]
    calibration_start = pd.Timestamp(splits["calibration"]["start"])
    validation_start = pd.Timestamp(splits["validation"]["start"])
    end = pd.Timestamp(splits["validation"]["end"])
    columns = [
        ID_COLUMN,
        DATE_COLUMN,
        HASH_COLUMN,
        NO_SHARED_COLUMN,
        NORMALIZED_COLUMN,
        CANONICAL_PRODUCT_COLUMN,
        *TARGETS,
        *COMPLETE_COLUMNS.values(),
    ]
    table = pq.read_table(
        input_path,
        columns=columns,
        filters=[
            (DATE_COLUMN, ">=", pd.Timestamp(splits["fit"]["start"]).date()),
            (DATE_COLUMN, "<", end.date()),
        ],
    )
    frame = table.to_pandas(categories=[CANONICAL_PRODUCT_COLUMN, "T1"])
    dates = pd.to_datetime(frame[DATE_COLUMN])
    frame["evaluation_split"] = np.select(
        [dates < calibration_start, dates < validation_start],
        ["fit", "calibration"],
        default="validation",
    )

    fit_hashes = set(frame.loc[frame["evaluation_split"] == "fit", HASH_COLUMN])
    frame["no_shared_text"] = True
    calibration = frame["evaluation_split"] == "calibration"
    validation = frame["evaluation_split"] == "validation"
    frame.loc[calibration, "no_shared_text"] = ~frame.loc[
        calibration,
        HASH_COLUMN,
    ].isin(fit_hashes)
    frame.loc[validation, "no_shared_text"] = frame.loc[
        validation,
        NO_SHARED_COLUMN,
    ]
    return frame.drop(columns=[DATE_COLUMN, HASH_COLUMN, NO_SHARED_COLUMN])


def sample_frames(
    frame: pd.DataFrame,
    sample_rows: dict[str, int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    samples = {}
    for offset, split in enumerate(("fit", "calibration", "validation")):
        available = frame.loc[frame["evaluation_split"] == split]
        requested = sample_rows[split]
        if requested > len(available):
            raise ValueError(f"M5 sample for {split} exceeds available rows.")
        if split == "fit":
            eligible_t1 = available[COMPLETE_COLUMNS["T1"]]
            required = (
                available.loc[eligible_t1]
                .groupby("T1", observed=True)
                .sample(n=1, random_state=seed)
            )
            remaining = available.drop(index=required.index).sample(
                n=requested - len(required),
                random_state=seed + offset,
            )
            sample = pd.concat([required, remaining])
        else:
            sample = available.sample(
                n=requested,
                random_state=seed + offset,
            )
        samples[split] = sample.sort_values(ID_COLUMN).reset_index(drop=True)
    return samples


def sample_contract(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, frame in frames.items():
        result[split] = {
            "rows": len(frame),
            "no_shared_text_rows": int(frame["no_shared_text"].sum()),
            "targets": {},
        }
        for target in TARGETS:
            complete = frame[COMPLETE_COLUMNS[target]].to_numpy(dtype=bool)
            no_shared = complete & frame["no_shared_text"].to_numpy(dtype=bool)
            if target == "T1":
                result[split]["targets"][target] = {
                    "complete": int(complete.sum()),
                    "no_shared_text": int(no_shared.sum()),
                    "classes_complete": int(frame.loc[complete, target].nunique()),
                    "classes_no_shared_text": int(frame.loc[no_shared, target].nunique()),
                }
            else:
                result[split]["targets"][target] = {
                    "complete": int(complete.sum()),
                    "no_shared_text": int(no_shared.sum()),
                    "positive_complete": int(
                        frame.loc[complete, target].to_numpy(dtype=bool).sum()
                    ),
                    "positive_no_shared_text": int(
                        frame.loc[no_shared, target].to_numpy(dtype=bool).sum()
                    ),
                }
    return result


def encode_bge(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], str | None]:
    torch = importlib.import_module("torch")
    sentence_transformers = importlib.import_module("sentence_transformers")
    sentence_transformer = sentence_transformers.SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("M5 requires a CUDA GPU.")
    torch.set_float32_matmul_precision("high")

    settings = config["bge"]
    model = sentence_transformer(
        settings["model"],
        revision=settings["revision"],
        device="cuda",
        local_files_only=True,
    )
    model.max_seq_length = settings["max_sequence_length"]
    revision = getattr(model[0].auto_model.config, "_commit_hash", settings["revision"])

    embeddings = {}
    for split, frame in frames.items():
        embeddings[split] = model.encode(
            frame[NORMALIZED_COLUMN].tolist(),
            batch_size=settings["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    return embeddings, revision


def load_cached_embeddings(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], str | None] | None:
    manifest_path = output_dir / "sample_manifest.parquet"
    metadata_path = output_dir / "metadata.json"
    embedding_paths = {
        split: output_dir / f"{split}_embeddings.npy" for split in frames
    }
    required_paths = [manifest_path, metadata_path, *embedding_paths.values()]
    if not all(path.exists() for path in required_paths):
        return None

    expected = pd.concat(
        [
            frame[[ID_COLUMN]].assign(sample_split=split)
            for split, frame in frames.items()
        ],
        ignore_index=True,
    ).astype(str)
    cached = pd.read_parquet(
        manifest_path,
        columns=[ID_COLUMN, "sample_split"],
    ).astype(str)
    if not cached.equals(expected):
        return None

    embeddings = {
        split: np.load(path) for split, path in embedding_paths.items()
    }
    if any(
        values.ndim != 2 or len(values) != len(frames[split])
        for split, values in embeddings.items()
    ):
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return embeddings, metadata.get("bge_revision")


def bge_feature_matrices(
    embeddings: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    product_encoder: Any,
) -> dict[str, np.ndarray]:
    matrices = {}
    for split, values in embeddings.items():
        product = product_encoder.transform(
            frames[split][[CANONICAL_PRODUCT_COLUMN]]
        ).toarray()
        matrices[split] = np.hstack([values, product]).astype(np.float32)
    return matrices


def add_comparison(
    tfidf_results: dict[str, Any],
    bge_results: dict[str, Any],
) -> dict[str, Any]:
    comparison = {}
    for target in TARGETS:
        primary = "macro_f1" if target == "T1" else "average_precision"
        tfidf_value = tfidf_results[target]["metrics"]["validation"][
            "no_shared_text"
        ][primary]
        bge_value = bge_results[target]["metrics"]["validation"][
            "no_shared_text"
        ][primary]
        comparison[target] = {
            "primary_metric": primary,
            "tfidf": tfidf_value,
            "bge": bge_value,
            "bge_improvement": bge_value - tfidf_value,
        }
    return comparison


def save_artifacts(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, np.ndarray],
    tfidf_vectorizer: Any,
    product_encoder: Any,
    tfidf_models: dict[str, Any],
    bge_models: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = pd.concat(
        [frame.assign(sample_split=split) for split, frame in frames.items()],
        ignore_index=True,
    ).drop(columns=[NORMALIZED_COLUMN])
    manifest.to_parquet(output_dir / "sample_manifest.parquet", index=False)
    for split, values in embeddings.items():
        np.save(output_dir / f"{split}_embeddings.npy", values)

    joblib.dump(tfidf_vectorizer, output_dir / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(product_encoder, output_dir / "product_encoder.joblib", compress=3)
    for target, model in tfidf_models.items():
        joblib.dump(model, output_dir / f"tfidf_{target.lower()}.joblib", compress=3)
    for target, model in bge_models.items():
        joblib.dump(model, output_dir / f"bge_{target.lower()}.joblib", compress=3)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_records(
    config: dict[str, Any],
    representations: dict[str, dict[str, Any]],
    revision: str | None,
) -> None:
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m5"
    for representation, results in representations.items():
        for target, result in results.items():
            parameters = {
                "representation": representation,
                "sample_fit_rows": config["bge"]["sample_rows"]["fit"],
                "sample_calibration_rows": config["bge"]["sample_rows"][
                    "calibration"
                ],
                "sample_validation_rows": config["bge"]["sample_rows"][
                    "validation"
                ],
                "model_artifact": config["paths"]["bge_artifacts"],
            }
            if representation == "bge_product":
                parameters.update(
                    {
                        "bge_model": config["bge"]["model"],
                        "bge_revision": revision or "unknown",
                        "max_sequence_length": config["bge"][
                            "max_sequence_length"
                        ],
                        "linear_max_iter": config["bge"]["linear_max_iter"],
                    }
                )
            if target != "T1":
                parameters["threshold"] = result["threshold"]

            record = build_run_record(
                config=config,
                run_name=f"m5-{target.lower()}-{representation}",
                stage="M5",
                target=target,
                split="sample_calibration_validation",
                view="no_shared_text_primary",
                features=config["features"]["initial"],
                parameters=parameters,
                metrics=flat_mlflow_metrics(result["metrics"]),
                artifacts=[config["paths"]["bge_report"]],
            )
            save_run_record(
                record,
                run_root / representation / target.lower() / "run.json",
            )


def main() -> None:
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    input_path = PROJECT_ROOT / config["paths"]["input_data"]
    input_contract_path = PROJECT_ROOT / config["paths"]["input_contract"]
    report_path = PROJECT_ROOT / config["paths"]["bge_report"]
    output_dir = PROJECT_ROOT / config["paths"]["bge_artifacts"]

    input_contract = json.loads(input_contract_path.read_text(encoding="utf-8"))
    source = load_source_rows(input_path, config)
    frames = sample_frames(source, config["bge"]["sample_rows"], seed)
    del source

    tfidf_vectorizer, product_encoder, tfidf_matrices = build_features(frames, config)
    tfidf_results, tfidf_models, _ = train_models(config, frames, tfidf_matrices)

    cached = load_cached_embeddings(output_dir, frames)
    if cached is None:
        embeddings, revision = encode_bge(frames, config)
    else:
        embeddings, revision = cached
        print("Reusing cached BGE embeddings for the matching sample.")

    bge_matrices = bge_feature_matrices(embeddings, frames, product_encoder)
    bge_config = deepcopy(config)
    bge_config["tfidf"]["max_iter"] = config["bge"]["linear_max_iter"]
    bge_results, bge_models, _ = train_models(bge_config, frames, bge_matrices)

    comparison = add_comparison(tfidf_results, bge_results)
    report = {
        "dvc_data_hash": input_contract["dvc_md5"],
        "split_version": config["evaluation"]["split_version"],
        "seed": seed,
        "sample": sample_contract(frames),
        "bge": {
            **config["bge"],
            "resolved_revision": revision,
            "embedding_dimensions": int(embeddings["fit"].shape[1]),
            "normalized": True,
        },
        "representations": {
            "tfidf_product": tfidf_results,
            "bge_product": bge_results,
        },
        "comparison": comparison,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "stage": "M5",
        "dvc_data_hash": input_contract["dvc_md5"],
        "split_version": config["evaluation"]["split_version"],
        "bge_model": config["bge"]["model"],
        "bge_revision": revision,
        "linear_max_iter": config["bge"]["linear_max_iter"],
        "sample_rows": config["bge"]["sample_rows"],
        "seed": seed,
    }
    save_artifacts(
        output_dir,
        frames,
        embeddings,
        tfidf_vectorizer,
        product_encoder,
        tfidf_models,
        bge_models,
        metadata,
    )
    save_records(
        config,
        {"tfidf_product": tfidf_results, "bge_product": bge_results},
        revision,
    )

    print("Validation without shared text on the fixed sample:")
    for target, values in comparison.items():
        print(
            f"{target}: TF-IDF={values['tfidf']:.6f}; "
            f"BGE={values['bge']:.6f}; change={values['bge_improvement']:+.6f}"
        )
    print(f"BGE revision: {revision}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
