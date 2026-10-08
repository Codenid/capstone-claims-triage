import unittest

from src.evaluation.comparison_table import COLUMNS, PROJECT_ROOT, build, verdict


class ComparisonTableTests(unittest.TestCase):
    def test_every_row_has_the_columns_and_an_existing_source(self):
        rows = build()

        self.assertGreater(len(rows), 50)
        for record in rows:
            self.assertEqual(list(record), COLUMNS)
            source = PROJECT_ROOT / record["fuente"]
            self.assertTrue(source.is_file(), record["fuente"])
            self.assertNotEqual(record["valor"], "", record["modelo"])

    def test_the_current_winners_appear_with_their_verdict(self):
        rows = {(r["etapa"], r["modelo"], r["conjunto"]): r for r in build()}

        key = ("M5F bloque B", "TabPFN-3.5 (PCA 100 + producto)", "calibracion_2024_q4")
        tabpfn = rows[key]
        self.assertEqual(tabpfn["decision"], "reemplaza")
        self.assertEqual(verdict(None), "")
        self.assertEqual(verdict(False), "no reemplaza")

    def test_daily_fulls_are_listed_with_their_baselines(self):
        built = build()
        rows = [r for r in built if r["etapa"].startswith("M9D")]

        models = {r["modelo"] for r in rows if r["rol"] == "candidato"}
        self.assertIn("D-A: binomial negativa diaria con día de semana", models)
        self.assertIn("D-E: Dirichlet-multinomial diaria", models)
        self.assertTrue(any(r["rol"] == "baseline" for r in rows))
        self.assertTrue(all(r["vista"] == "40 patrones, diario" for r in rows))
        alerts = [r for r in built if r["etapa"] == "M11D"]
        self.assertTrue(alerts)
        candidates = [r for r in alerts if r["rol"] == "candidato"]
        self.assertTrue(all(r["decision"] == "no adoptada" for r in candidates))


if __name__ == "__main__":
    unittest.main()
