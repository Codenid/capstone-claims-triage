"""Generate complete BGE embeddings with resumable checkpoints on Khipu."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.normalize_text import HASH_COLUMN, NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.semantic_space import ID_COLUMN, load_dvc_hash
from src.models.tfidf_models import DATE_COLUMN

SPLITS = ("fit", "calibration", "validation")
STOP_REQUESTED = False


def request_stop(_signal_number: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def split_filter(
    split: str,
    config: dict[str, Any],
    eligible_column: str,
) -> Any:
    period = config["evaluation"]["splits"][split]
    return (
        (ds.field(DATE_COLUMN) >= pa.scalar(pd.Timestamp(period["start"]).date()))
        & (ds.field(DATE_COLUMN) < pa.scalar(pd.Timestamp(period["end"]).date()))
        & (ds.field(eligible_column) == True)  # noqa: E712
    )


def config_fingerprint(config: dict[str, Any]) -> str:
    import hashlib

    relevant = {
        "bge": config["bge"],
        "bge_full": config["bge_full"],
        "splits": config["evaluation"]["splits"],
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initial_state(
    config: dict[str, Any],
    input_dvc_hash: str,
    sample_dvc_hash: str,
) -> dict[str, Any]:
    return {
        "contract_version": config["bge_full"]["contract_version"],
        "config_sha256": config_fingerprint(config),
        "input_dvc_hash": input_dvc_hash,
        "sample_dvc_hash": sample_dvc_hash,
        "model_revision": config["bge"]["revision"],
        "expected_rows": config["bge_full"]["expected_rows"],
        "progress": {split: 0 for split in SPLITS},
        "reused_rows": {split: 0 for split in SPLITS},
        "generated_rows": {split: 0 for split in SPLITS},
        "encoding_seconds": 0.0,
        "resume_count": 0,
        "status": "running",
    }


def validate_state(
    state: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    immutable = (
        "contract_version",
        "config_sha256",
        "input_dvc_hash",
        "sample_dvc_hash",
        "model_revision",
        "expected_rows",
    )
    for name in immutable:
        if state.get(name) != expected.get(name):
            raise ValueError(f"M8A checkpoint does not match current {name}.")


def save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_or_create_state(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        save_state(path, expected)
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    validate_state(state, expected)
    state["resume_count"] += 1
    save_state(path, state)
    return state


def load_sample_embeddings(
    sample_dir: Path,
    expected_rows: dict[str, int],
) -> tuple[dict[str, dict[str, int]], dict[str, np.ndarray]]:
    manifest = pd.read_parquet(
        sample_dir / "sample_manifest.parquet",
        columns=[ID_COLUMN, "sample_split"],
    )
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("M8A source sample contains duplicate IDs.")

    indices = {}
    embeddings = {}
    for split in SPLITS:
        frame = manifest.loc[manifest["sample_split"] == split].reset_index(drop=True)
        if len(frame) != expected_rows[split]:
            raise ValueError(f"M8A source sample count does not match {split}.")
        indices[split] = {
            complaint_id: row
            for row, complaint_id in enumerate(frame[ID_COLUMN].astype(str))
        }
        embeddings[split] = np.load(
            sample_dir / f"{split}_embeddings.npy",
            mmap_mode="r",
        )
    return indices, embeddings


def merge_embedding_batch(
    complaint_ids: list[str],
    narratives: list[str],
    sample_index: dict[str, int],
    sample_embeddings: np.ndarray,
    encode: Callable[[list[str]], np.ndarray],
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    result = np.empty((len(complaint_ids), dimensions), dtype=np.float32)
    reused = np.array(
        [complaint_id in sample_index for complaint_id in complaint_ids],
        dtype=bool,
    )
    reused_positions = np.flatnonzero(reused)
    if len(reused_positions):
        source_positions = [sample_index[complaint_ids[row]] for row in reused_positions]
        result[reused_positions] = sample_embeddings[source_positions]

    missing_positions = np.flatnonzero(~reused)
    elapsed = 0.0
    if len(missing_positions):
        missing_texts = [narratives[row] for row in missing_positions]
        if any(not isinstance(text, str) or not text for text in missing_texts):
            raise ValueError("M8A found an empty eligible narrative.")
        started = time.perf_counter()
        generated = encode(missing_texts)
        elapsed = time.perf_counter() - started
        if generated.shape != (len(missing_positions), dimensions):
            raise ValueError("M8A encoder returned an unexpected shape.")
        result[missing_positions] = generated.astype(np.float32, copy=False)
    return result, reused, elapsed


def load_bge(config: dict[str, Any]) -> tuple[Any, Any, str]:
    torch = importlib.import_module("torch")
    sentence_transformers = importlib.import_module("sentence_transformers")
    if not torch.cuda.is_available():
        raise RuntimeError("M8A requires a CUDA GPU.")
    torch.set_float32_matmul_precision("high")

    settings = config["bge"]
    model = sentence_transformers.SentenceTransformer(
        settings["model"],
        revision=settings["revision"],
        device="cuda",
        local_files_only=True,
    )
    model.max_seq_length = settings["max_sequence_length"]
    revision = getattr(model[0].auto_model.config, "_commit_hash", settings["revision"])
    if revision != settings["revision"]:
        raise ValueError("M8A resolved BGE revision does not match configuration.")
    return model, torch, revision


def make_encoder(model: Any, batch_size: int) -> Callable[[list[str]], np.ndarray]:
    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    return encode


def open_output_array(
    path: Path,
    rows: int,
    dimensions: int,
) -> np.memmap:
    if path.exists():
        values = cast(np.memmap, np.load(path, mmap_mode="r+"))
        if values.shape != (rows, dimensions) or values.dtype != np.float32:
            raise ValueError(f"M8A staging array is invalid: {path}")
        return values
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, dimensions),
    )


def write_manifest_part(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def process_split(
    split: str,
    dataset: ds.Dataset,
    config: dict[str, Any],
    staging_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    sample_index: dict[str, int],
    sample_embeddings: np.ndarray,
    encode: Callable[[list[str]], np.ndarray],
) -> None:
    settings = config["bge_full"]
    expected_rows = settings["expected_rows"][split]
    dimensions = settings["embedding_dimensions"]
    output = open_output_array(
        staging_dir / f"{split}_embeddings.npy",
        expected_rows,
        dimensions,
    )
    parts_dir = staging_dir / "manifest_parts" / split
    parts_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        ID_COLUMN,
        DATE_COLUMN,
        NORMALIZED_COLUMN,
        HASH_COLUMN,
        CANONICAL_PRODUCT_COLUMN,
    ]
    scanner = dataset.scanner(
        columns=columns,
        filter=split_filter(split, config, settings["eligible_column"]),
        batch_size=settings["source_batch_rows"],
        use_threads=True,
    )

    def commit_chunk(table: pa.Table) -> None:
        frame = table.to_pandas()
        complaint_ids = frame[ID_COLUMN].astype(str).tolist()
        narratives = frame[NORMALIZED_COLUMN].tolist()
        values, reused, encoding_seconds = merge_embedding_batch(
            complaint_ids,
            narratives,
            sample_index,
            sample_embeddings,
            encode,
            dimensions,
        )
        start = state["progress"][split]
        stop = start + len(frame)
        output[start:stop] = values
        output.flush()

        manifest_part = pd.DataFrame(
            {
                ID_COLUMN: complaint_ids,
                DATE_COLUMN: frame[DATE_COLUMN],
                HASH_COLUMN: frame[HASH_COLUMN].astype(str),
                CANONICAL_PRODUCT_COLUMN: frame[CANONICAL_PRODUCT_COLUMN].astype(str),
                "split": split,
                "embedding_row": np.arange(start, stop, dtype=np.int64),
                "reused_m5": reused,
            }
        )
        part_path = parts_dir / f"part-{start:09d}-{stop:09d}.parquet"
        write_manifest_part(part_path, manifest_part)

        state["progress"][split] = stop
        state["reused_rows"][split] += int(reused.sum())
        state["generated_rows"][split] += int((~reused).sum())
        state["encoding_seconds"] += encoding_seconds
        save_state(state_path, state)

        if STOP_REQUESTED:
            print(f"Checkpoint saved after {split} row {stop}.")
            raise SystemExit(0)

    source_rows_seen = 0
    skip_remaining = state["progress"][split]
    pending: pa.Table | None = None
    for record_batch in scanner.to_batches():
        table = pa.Table.from_batches([record_batch])
        source_rows_seen += len(table)
        if skip_remaining >= len(table):
            skip_remaining -= len(table)
            continue
        if skip_remaining:
            table = table.slice(skip_remaining)
            skip_remaining = 0
        if not len(table):
            continue
        pending = table if pending is None else pa.concat_tables([pending, table])
        while len(pending) >= settings["arrow_batch_rows"]:
            commit_chunk(pending.slice(0, settings["arrow_batch_rows"]))
            pending = pending.slice(settings["arrow_batch_rows"])

    if pending is not None and len(pending):
        commit_chunk(pending)
    if source_rows_seen != expected_rows or state["progress"][split] != expected_rows:
        raise ValueError(
            f"M8A expected {expected_rows} {split} rows but found "
            f"{source_rows_seen}."
        )


def combine_manifest(staging_dir: Path) -> Path:
    output_path = staging_dir / "manifest.parquet"
    writer = None
    try:
        for split in SPLITS:
            parts = sorted((staging_dir / "manifest_parts" / split).glob("*.parquet"))
            for part in parts:
                table = pq.read_table(part)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("M8A did not create manifest parts.")
    return output_path


def validate_outputs(
    staging_dir: Path,
    config: dict[str, Any],
    sample_indices: dict[str, dict[str, int]],
    sample_embeddings: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    settings = config["bge_full"]
    manifest_path = staging_dir / "manifest.parquet"
    manifest_ids = pd.read_parquet(manifest_path, columns=[ID_COLUMN])
    if manifest_ids[ID_COLUMN].duplicated().any():
        raise ValueError("M8A final manifest contains duplicate IDs.")
    if len(manifest_ids) != sum(settings["expected_rows"].values()):
        raise ValueError("M8A final manifest row count is incorrect.")
    del manifest_ids

    validation = {}
    for split in SPLITS:
        values = np.load(staging_dir / f"{split}_embeddings.npy", mmap_mode="r")
        expected_shape = (
            settings["expected_rows"][split],
            settings["embedding_dimensions"],
        )
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError(f"M8A final array is invalid for {split}.")

        maximum_norm_error = 0.0
        for start in range(0, len(values), settings["arrow_batch_rows"]):
            batch = np.asarray(values[start : start + settings["arrow_batch_rows"]])
            if not np.isfinite(batch).all():
                raise ValueError(f"M8A {split} embeddings contain non-finite values.")
            maximum_norm_error = max(
                maximum_norm_error,
                float(np.abs(np.linalg.norm(batch, axis=1) - 1.0).max()),
            )
        if maximum_norm_error > settings["normalization_tolerance"]:
            raise ValueError(f"M8A {split} embeddings are not normalized.")

        reused = pd.read_parquet(
            manifest_path,
            columns=[ID_COLUMN, "embedding_row", "reused_m5"],
            filters=[("split", "=", split), ("reused_m5", "=", True)],
        )
        if len(reused) != settings["expected_reused_rows"][split]:
            raise ValueError(f"M8A reused count is incorrect for {split}.")
        for start in range(0, len(reused), 1024):
            batch = reused.iloc[start : start + 1024]
            output_rows = batch["embedding_row"].to_numpy(dtype=np.int64)
            source_rows = np.array(
                [sample_indices[split][value] for value in batch[ID_COLUMN].astype(str)],
                dtype=np.int64,
            )
            if not np.array_equal(
                values[output_rows],
                sample_embeddings[split][source_rows],
            ):
                raise ValueError(f"M8A reused embeddings differ for {split}.")

        validation[split] = {
            "rows": len(values),
            "dimensions": values.shape[1],
            "dtype": str(values.dtype),
            "maximum_norm_error": maximum_norm_error,
            "bytes": (staging_dir / f"{split}_embeddings.npy").stat().st_size,
        }
    return validation


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    total_generated = sum(report["generated_rows"].values())
    record = build_run_record(
        config=config,
        run_name="m8a-bge-full",
        stage="M8A",
        target="semantic_embeddings",
        split="fit_calibration_validation",
        view="complete_eligible_narratives",
        features=["Consumer complaint narrative normalized"],
        parameters={
            "model": config["bge"]["model"],
            "revision": config["bge"]["revision"],
            "dimensions": config["bge_full"]["embedding_dimensions"],
            "gpu_batch_size": config["bge"]["batch_size"],
            "source_batch_rows": config["bge_full"]["source_batch_rows"],
            "arrow_batch_rows": config["bge_full"]["arrow_batch_rows"],
            "source_sample_dvc_hash": config["bge_full"]["source_dvc_hash"],
            "model_artifact": config["paths"]["bge_full_artifacts"],
        },
        metrics={
            "total_rows": float(sum(report["rows"].values())),
            "reused_rows": float(sum(report["reused_rows"].values())),
            "generated_rows": float(total_generated),
            "generated_rows_per_second": float(report["generated_rows_per_second"]),
            "elapsed_seconds": float(report["elapsed_seconds"]),
            "maximum_norm_error": float(
                max(item["maximum_norm_error"] for item in report["validation"].values())
            ),
        },
        artifacts=[config["paths"]["bge_full_report"]],
    )
    path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m8a"
        / "bge_full"
        / "run.json"
    )
    save_run_record(record, path)


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    set_seed(config["experiment"]["seed"])
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is not None:
        signal.signal(sigusr1, request_stop)

    settings = config["bge_full"]
    output_dir = PROJECT_ROOT / config["paths"]["bge_full_artifacts"]
    staging_dir = PROJECT_ROOT / config["paths"]["bge_full_staging"]
    report_path = PROJECT_ROOT / config["paths"]["bge_full_report"]
    if (output_dir / "_SUCCESS").exists():
        print(f"M8A is already complete: {output_dir.relative_to(PROJECT_ROOT)}")
        return
    if output_dir.exists():
        raise ValueError("M8A final output exists without _SUCCESS.")
    staging_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    input_contract = json.loads(
        (PROJECT_ROOT / config["paths"]["input_contract"]).read_text(encoding="utf-8")
    )
    sample_dvc_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["bge_artifacts_dvc"]
    )
    if sample_dvc_hash != settings["source_dvc_hash"]:
        raise ValueError("M8A source sample DVC hash is not frozen.")

    expected_state = initial_state(config, input_contract["dvc_md5"], sample_dvc_hash)
    state_path = staging_dir / "checkpoint.json"
    state = load_or_create_state(state_path, expected_state)
    sample_indices, sample_embeddings = load_sample_embeddings(
        PROJECT_ROOT / config["paths"]["bge_artifacts"],
        settings["expected_reused_rows"],
    )

    model, torch, resolved_revision = load_bge(config)
    encode = make_encoder(model, config["bge"]["batch_size"])
    dataset = ds.dataset(
        PROJECT_ROOT / config["paths"]["input_data"],
        format="parquet",
    )
    for split in SPLITS:
        process_split(
            split,
            dataset,
            config,
            staging_dir,
            state,
            state_path,
            sample_indices[split],
            sample_embeddings[split],
            encode,
        )

    manifest_path = combine_manifest(staging_dir)
    validation = validate_outputs(
        staging_dir,
        config,
        sample_indices,
        sample_embeddings,
    )
    elapsed_seconds = time.perf_counter() - started
    total_generated = sum(state["generated_rows"].values())
    report = {
        "stage": "M8A",
        "contract_version": settings["contract_version"],
        "input_dvc_hash": input_contract["dvc_md5"],
        "source_sample_dvc_hash": sample_dvc_hash,
        "model": config["bge"]["model"],
        "revision": resolved_revision,
        "rows": settings["expected_rows"],
        "reused_rows": state["reused_rows"],
        "generated_rows": state["generated_rows"],
        "validation": validation,
        "manifest_rows": pq.read_metadata(manifest_path).num_rows,
        "resume_count": state["resume_count"],
        "encoding_seconds": state["encoding_seconds"],
        "elapsed_seconds": elapsed_seconds,
        "generated_rows_per_second": (
            total_generated / state["encoding_seconds"]
            if state["encoding_seconds"]
            else 0.0
        ),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "maximum_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        },
    }
    (staging_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    state["status"] = "complete"
    save_state(state_path, state)
    shutil.rmtree(staging_dir / "manifest_parts")
    (staging_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    staging_dir.replace(output_dir)
    save_offline_record(config, report)

    print(f"Rows: {sum(report['rows'].values()):,}")
    print(f"Reused: {sum(report['reused_rows'].values()):,}")
    print(f"Generated: {sum(report['generated_rows'].values()):,}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
