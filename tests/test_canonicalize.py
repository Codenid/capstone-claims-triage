from __future__ import annotations

import unittest

import pyarrow as pa

from src.data.canonicalize import cargar_taxonomia, canonicalizar_tabla, mapas


class CanonicalizeTest(unittest.TestCase):
    def test_taxonomy_has_unique_complete_declared_maps(self) -> None:
        issues, products, pairs = mapas(cargar_taxonomia())
        self.assertEqual(len(issues), 173)
        self.assertEqual(len(set(issues)), 173)
        self.assertEqual(len(set(products) | {p.split("\x1f")[0] for p in pairs}), 21)

    def test_canonicalizes_direct_and_composite_product_rules(self) -> None:
        tabla = pa.table(
            {
                "issue": ["Incorrect information on your report", "Problem with cash advance"],
                "product": ["Credit reporting", "Credit card or prepaid card"],
                "sub_product": ["SIN_DATO", "General-purpose prepaid card"],
            }
        )
        salida = canonicalizar_tabla(tabla, cargar_taxonomia())
        self.assertEqual(
            salida["producto_canonico"].to_pylist(),
            ["REPORTE_CREDITO_Y_MONITOREO", "TARJETA_PREPAGO"],
        )
        self.assertEqual(
            salida["motivo_canonico"].to_pylist(),
            ["REPORTE_CON_INFORMACION_INCORRECTA", "OTRO_PROBLEMA_TRANSACCIONAL_O_DE_SERVICIO"],
        )
        self.assertEqual(salida.num_rows, tabla.num_rows)

    def test_unknown_issue_fails_instead_of_falling_back(self) -> None:
        tabla = pa.table(
            {"issue": ["New issue"], "product": ["Mortgage"], "sub_product": ["Other"]}
        )
        with self.assertRaisesRegex(ValueError, "issue"):
            canonicalizar_tabla(tabla, cargar_taxonomia())


if __name__ == "__main__":
    unittest.main()
