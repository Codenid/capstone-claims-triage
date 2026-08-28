from __future__ import annotations

import unittest

import pyarrow as pa
import yaml

from src.data.typing import CONFIG_PATH, SCHEMA_SALIDA, transformar_lote


class TypingTest(unittest.TestCase):
    def test_transform_preserves_rows_and_derives_contract(self) -> None:
        with CONFIG_PATH.open(encoding="utf-8") as archivo:
            config = yaml.safe_load(archivo)
        lote = pa.RecordBatch.from_pydict(
            {
                "Date received": ["2024-01-02", "2025-07-01"],
                "Product": ["Credit card", "Mortgage"],
                "Sub-product": [None, "Conventional home mortgage"],
                "Issue": ["Billing dispute", "Applying for a mortgage"],
                "Sub-issue": [None, "Application denied"],
                "Consumer complaint narrative": ["First complaint", "Second complaint"],
                "Company public response": [None, "Company believes it acted appropriately"],
                "Company": ["Bank A", "Bank B"],
                "State": ["NY", "CA"],
                "ZIP code": ["10001", None],
                "Tags": ["Older American, Servicemember", None],
                "Submitted via": ["Web", "Web"],
                "Date sent to company": ["2024-01-03", "2025-07-02"],
                "Company response to consumer": [
                    "Closed with monetary relief",
                    "Closed with explanation",
                ],
                "Timely response?": ["Yes", "No"],
                "Complaint ID": ["10", "11"],
            }
        )

        tabla = transformar_lote(lote, config)

        self.assertEqual(tabla.schema, SCHEMA_SALIDA)
        self.assertEqual(tabla.num_rows, 2)
        self.assertEqual(tabla["complaint_id"].to_pylist(), [10, 11])
        self.assertEqual(tabla["is_servicemember"].to_pylist(), [True, False])
        self.assertEqual(tabla["is_older_adult"].to_pylist(), [True, False])
        self.assertEqual(tabla["missing_sub_issue"].to_pylist(), [True, False])
        self.assertEqual(tabla["target_t2_relief"].to_pylist(), [1, 0])
        self.assertEqual(tabla["target_t3_monetary"].to_pylist(), [1, 0])
        self.assertEqual(tabla["target_t4_late"].to_pylist(), [0, 1])

    def test_unknown_domain_fails(self) -> None:
        with CONFIG_PATH.open(encoding="utf-8") as archivo:
            config = yaml.safe_load(archivo)
        columnas = {nombre: [None] for nombre in config["source_columns"]}
        columnas.update(
            {
                "Date received": ["2024-01-01"],
                "Date sent to company": ["2024-01-02"],
                "Consumer complaint narrative": ["Complaint"],
                "Submitted via": ["Phone"],
                "Complaint ID": ["1"],
            }
        )
        lote = pa.RecordBatch.from_pydict(columnas)
        with self.assertRaisesRegex(ValueError, "Submitted via"):
            transformar_lote(lote, config)


if __name__ == "__main__":
    unittest.main()
