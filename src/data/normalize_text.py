"""Normalize complaint narratives and create stable SHA-256 hashes."""

from hashlib import sha256
from pathlib import Path
import re
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

NORMALIZER_VERSION = "text_normalizer_v1"
NARRATIVE_COLUMN = "Consumer complaint narrative"
NORMALIZED_COLUMN = "Consumer complaint narrative normalized"
HASH_COLUMN = "Consumer complaint narrative SHA-256"

_REDACTION_PATTERN = re.compile(r"\bx{2,}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply the approved normalization rules in their fixed order."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = _REDACTION_PATTERN.sub("<redacted>", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def hash_text(text: str) -> str:
    """Return the SHA-256 hash of UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


def add_normalized_columns(table: pa.Table) -> pa.Table:
    """Append normalized narratives and their hashes without removing rows."""
    narratives = table[NARRATIVE_COLUMN].combine_chunks().to_pylist()
    normalized = [
        normalize_text(text) if text is not None else None for text in narratives
    ]
    hashes = [hash_text(text) if text is not None else None for text in normalized]

    result = table.append_column(
        NORMALIZED_COLUMN,
        pa.array(normalized, type=pa.large_string()),
    )
    result = result.append_column(HASH_COLUMN, pa.array(hashes, type=pa.string()))

    metadata = dict(result.schema.metadata or {})
    metadata[b"text_normalizer"] = NORMALIZER_VERSION.encode("utf-8")
    return result.replace_schema_metadata(metadata)


def prepare(input_path: Path, output_path: Path) -> None:
    """Normalize a Parquet file in batches and write a new Parquet."""
    source = pq.ParquetFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=50_000):
            normalized = add_normalized_columns(pa.Table.from_batches([batch]))
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    normalized.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(normalized, row_group_size=50_000)
            rows += normalized.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("The input file is empty.")
    if rows != source.metadata.num_rows:
        raise ValueError("Normalization changed the number of rows.")

    print(f"Rows: {rows:,}")
    print(f"Columns: {len(source.schema.names) + 2}")
    print(f"Normalizer: {NORMALIZER_VERSION}")
    print(f"Output: {output_path}")


def main() -> None:
    prepare(
        Path("data/interim/typed.parquet"),
        Path("data/interim/normalized.parquet"),
    )


if __name__ == "__main__":
    main()
