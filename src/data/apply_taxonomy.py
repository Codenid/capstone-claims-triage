"""Apply the frozen proposed taxonomy to the normalized complaints."""

from csv import DictReader
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TAXONOMY_VERSION = "taxonomy_v1_proposed"
TAXONOMY_STATUS = "proposed_no_business_review"
FREEZE_DATE = date(2025, 6, 30)
UNKNOWN_CATEGORY = "__UNKNOWN__"

CANONICAL_PRODUCT_COLUMN = "Product canonical"
CANONICAL_ISSUE_COLUMN = "Issue canonical"
KNOWN_PRODUCT_COLUMN = "known_product"
KNOWN_ISSUE_COLUMN = "known_issue"
KNOWN_PAIR_COLUMN = "known_product_issue_pair"


def load_mapping(path: Path, raw_column: str, canonical_column: str) -> dict[str, str]:
    """Load a one-to-one explicit mapping and reject duplicate inputs."""
    mapping = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in DictReader(file):
            raw_value = row[raw_column]
            if raw_value in mapping:
                raise ValueError(f"Duplicate mapping for {raw_value!r} in {path}.")
            mapping[raw_value] = row[canonical_column]
    return mapping


def load_pairs(path: Path) -> set[tuple[str, str]]:
    """Load the valid raw Product-Issue pairs frozen through 2025-H1."""
    with path.open(encoding="utf-8", newline="") as file:
        return {
            (row["raw_product"], row["raw_issue"])
            for row in DictReader(file)
        }


def apply_taxonomy(
    table: pa.Table,
    product_map: dict[str, str],
    issue_map: dict[str, str],
    valid_pairs: set[tuple[str, str]],
) -> pa.Table:
    """Append canonical categories and flags without changing original columns."""
    dates = table["Date received"].combine_chunks().to_pylist()
    products = table["Product"].combine_chunks().to_pylist()
    issues = table["Issue"].combine_chunks().to_pylist()

    canonical_products = []
    canonical_issues = []
    known_products = []
    known_issues = []
    known_pairs = []

    for received, product, issue in zip(dates, products, issues):
        product_is_known = product in product_map
        issue_is_known = issue in issue_map
        pair_is_known = (product, issue) in valid_pairs

        if received is None:
            raise ValueError("Date received cannot be null.")
        if received <= FREEZE_DATE and not (
            product_is_known and issue_is_known and pair_is_known
        ):
            raise ValueError(
                "Unmapped category or invalid Product-Issue pair before the "
                f"taxonomy freeze date: {product!r}, {issue!r}."
            )

        canonical_products.append(
            product_map[product] if product_is_known else UNKNOWN_CATEGORY
        )
        canonical_issues.append(issue_map[issue] if issue_is_known else None)
        known_products.append(product_is_known)
        known_issues.append(issue_is_known)
        known_pairs.append(pair_is_known)

    result = table.append_column(
        CANONICAL_PRODUCT_COLUMN,
        pa.array(canonical_products, type=pa.large_string()),
    )
    result = result.append_column(
        CANONICAL_ISSUE_COLUMN,
        pa.array(canonical_issues, type=pa.large_string()),
    )
    result = result.append_column(KNOWN_PRODUCT_COLUMN, pa.array(known_products))
    result = result.append_column(KNOWN_ISSUE_COLUMN, pa.array(known_issues))
    result = result.append_column(KNOWN_PAIR_COLUMN, pa.array(known_pairs))

    metadata = dict(result.schema.metadata or {})
    metadata[b"taxonomy_version"] = TAXONOMY_VERSION.encode("utf-8")
    metadata[b"taxonomy_status"] = TAXONOMY_STATUS.encode("utf-8")
    metadata[b"taxonomy_freeze_date"] = FREEZE_DATE.isoformat().encode("utf-8")
    return result.replace_schema_metadata(metadata)


def prepare(
    input_path: Path,
    output_path: Path,
    product_map_path: Path,
    issue_map_path: Path,
    pairs_path: Path,
) -> None:
    """Apply the taxonomy in batches and write a new Parquet."""
    product_map = load_mapping(
        product_map_path,
        "raw_product",
        "canonical_product",
    )
    issue_map = load_mapping(issue_map_path, "raw_issue", "canonical_issue")
    valid_pairs = load_pairs(pairs_path)
    source = pq.ParquetFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0
    unknown_products = 0
    unknown_issues = 0
    unknown_pairs = 0
    try:
        for batch in source.iter_batches(batch_size=50_000):
            mapped = apply_taxonomy(pa.Table.from_batches([batch]), product_map, issue_map, valid_pairs)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    mapped.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(mapped, row_group_size=50_000)
            rows += mapped.num_rows
            unknown_products += mapped[KNOWN_PRODUCT_COLUMN].to_pylist().count(False)
            unknown_issues += mapped[KNOWN_ISSUE_COLUMN].to_pylist().count(False)
            unknown_pairs += mapped[KNOWN_PAIR_COLUMN].to_pylist().count(False)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("The input file is empty.")
    if rows != source.metadata.num_rows:
        raise ValueError("Taxonomy mapping changed the number of rows.")

    print(f"Rows: {rows:,}")
    print(f"Columns: {len(source.schema.names) + 5}")
    print(f"Taxonomy: {TAXONOMY_VERSION}")
    print(f"Unknown products after freeze: {unknown_products:,}")
    print(f"Unknown issues after freeze: {unknown_issues:,}")
    print(f"Unknown Product-Issue pairs after freeze: {unknown_pairs:,}")
    print(f"Output: {output_path}")


def main() -> None:
    prepare(
        Path("data/interim/normalized.parquet"),
        Path("data/interim/taxonomy.parquet"),
        Path("configs/product_taxonomy_v1.csv"),
        Path("configs/issue_taxonomy_v1.csv"),
        Path("configs/valid_product_issue_pairs_v1.csv"),
    )


if __name__ == "__main__":
    main()
