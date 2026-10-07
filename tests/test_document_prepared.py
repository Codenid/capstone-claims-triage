import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.document_prepared import (
    DESCRIPTIONS_PATH,
    PARAMS_PATH,
    build_report,
    dictionary,
    load_params,
)

COLUMNS = {
    "type_data": ["Complaint ID", "Product", "Issue", "Consumer complaint narrative"],
    "normalize_text": ["Consumer complaint narrative SHA-256"],
    "apply_taxonomy": [
        "Product canonical",
        "Issue canonical",
        "known_product_issue_pair",
    ],
    "build_targets": [
        "period",
        "T1",
        "T2",
        "T3",
        "T4",
        "known_T2",
        "known_T4",
        "previously_reviewed_id",
        "excluded_by_reviewed_text_group",
        "holdout_status",
        "no_shared_reference_text",
        "eligible_T1_complete",
    ],
    "finalize_prepared": [],
}


def small_table() -> pa.Table:
    return pa.table(
        {
            "Complaint ID": ["1", "2", "3", "4"],
            "Product": ["Mortgage", "Mortgage", "Credit card", "Credit card"],
            "Issue": ["Closing", "Closing", "Fees", "Other"],
            "Consumer complaint narrative": ["a text", "a text", "other", "more"],
            "Consumer complaint narrative SHA-256": ["h1", "h1", "h2", "h3"],
            "Product canonical": ["Mortgage", "Mortgage", "Credit card", "Credit card"],
            "Issue canonical": ["Closing", "Closing", "Fees", "Other"],
            "known_product_issue_pair": [True, True, True, False],
            "period": ["train_2023_2024"] * 2 + ["validation_2025_h1"] * 2,
            "T1": ["Closing", "Closing", "Fees", "Other"],
            "T2": [True, False, True, None],
            "T3": [False, False, True, None],
            "T4": [False, False, False, True],
            "known_T2": [True, True, True, False],
            "known_T4": [True] * 4,
            "previously_reviewed_id": [False, False, False, True],
            "excluded_by_reviewed_text_group": [False] * 4,
            "holdout_status": ["not_holdout"] * 3 + ["blocked"],
            "no_shared_reference_text": [True, False, True, True],
            "eligible_T1_complete": [True, True, True, False],
        }
    )


def params() -> dict:
    roles = {
        "identificador": ["Complaint ID"],
        "predictora": ["Product", "Product canonical", "Consumer complaint narrative"],
        "posterior_al_evento": ["Issue", "Issue canonical"],
        "objetivo": ["T1", "T2", "T3", "T4"],
    }
    control = [
        column
        for columns in COLUMNS.values()
        for column in columns
        if column not in sum(roles.values(), [])
    ]
    return {
        "columnas_por_etapa": COLUMNS,
        "descripcion_etapas": {stage: stage for stage in COLUMNS},
        "roles": roles | {"control_de_evaluacion": control},
        "columnas_sin_ejemplo": ["Consumer complaint narrative"],
        "ejemplo_max_caracteres": 6,
    }


def lock() -> dict:
    return {
        stage: {"outs": [{"path": f"data/{stage}.parquet", "md5": stage, "size": 1}]}
        for stage in COLUMNS
    }


class DocumentPreparedTests(unittest.TestCase):
    def test_dictionary_counts_describes_and_hides_narratives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.parquet"
            pq.write_table(small_table(), path)
            with pq.ParquetFile(path) as table:
                rows = dictionary(table, params(), {"T2": "Alguna solución"})

        by_name = {row["columna"]: row for row in rows}
        self.assertEqual(len(rows), small_table().num_columns)
        self.assertEqual(by_name["T2"]["descripcion"], "Alguna solución")
        self.assertEqual(by_name["T2"]["no_nulos"], 3)
        self.assertEqual(by_name["T2"]["pct_nulos"], 25.0)
        self.assertEqual(by_name["Product"]["valores_unicos"], 2)
        self.assertEqual(by_name["Product"]["ejemplo"], "Mortg…")
        narrative = by_name["Consumer complaint narrative"]
        self.assertEqual(narrative["ejemplo"], "(texto libre)")
        self.assertEqual(by_name["period"]["etapa"], "build_targets")
        self.assertEqual(by_name["Issue"]["rol"], "posterior_al_evento")
        self.assertTrue(by_name["T1"]["descripcion"].startswith("PENDIENTE"))

    def test_report_measures_the_findings_on_the_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.parquet"
            pq.write_table(small_table(), path)
            with pq.ParquetFile(path) as table:
                rows = dictionary(table, params(), {})
                report = build_report(table, rows, params(), lock())

        self.assertEqual(report["tabla"]["filas"], 4)
        self.assertEqual([stage["etapa"] for stage in report["etapas"]], list(COLUMNS))
        self.assertEqual(report["etapas"][-1]["columnas_salida"], 20)
        problems = {item["columnas"][0]: item["cifra"] for item in report["problemas"]}
        shared = problems["Consumer complaint narrative SHA-256"]
        self.assertEqual(shared["textos_normalizados_distintos"], 3)
        self.assertEqual(shared["filas_que_comparten_texto"], 2)
        self.assertEqual(problems["period"]["filas_por_periodo"]["train_2023_2024"], 2)
        self.assertEqual(problems["T3"]["pct_positivos_T2_conocidos"], 66.667)
        reviewed = problems["previously_reviewed_id"]
        self.assertEqual(reviewed["identificadores_revisados"], 1)
        self.assertEqual(report["diccionario"]["cobertura_pct"], 0.0)
        json.dumps(report)

    def test_declared_columns_and_descriptions_cover_the_contract(self):
        declared = load_params(PARAMS_PATH)["columnas_por_etapa"]
        columns = [column for stage in declared.values() for column in stage]
        descriptions = yaml.safe_load(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len(columns), 44)
        self.assertEqual(len(set(columns)), 44)
        self.assertEqual(set(descriptions), set(columns))


if __name__ == "__main__":
    unittest.main()
