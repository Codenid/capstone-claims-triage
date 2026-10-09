import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "reports" / "informe" / "facts.py"


def load_facts_module():
    spec = importlib.util.spec_from_file_location("informe_facts", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InformeFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = load_facts_module()
        cls.F = cls.facts.build()

    def test_every_section_loads_from_the_result_files(self):
        F = self.F
        self.assertEqual(
            set(F),
            {"datos", "clasificadores", "semantico", "semanal", "composicion",
             "alertas", "diario", "catalogo", "tabla"},
        )
        self.assertGreater(F["datos"]["rows"], 3_000_000)
        self.assertEqual(len(F["catalogo"]), 40)
        self.assertGreater(len(F["tabla"]), 100)

    def test_the_numbers_the_text_cites_are_finite(self):
        F = self.F
        cited = [
            F["clasificadores"]["tabpfn"]["macro_f1"],
            F["clasificadores"]["tasks"]["T1"]["rule_calibration"],
            F["clasificadores"]["tasks"]["T2"]["final_model"],
            F["semantico"]["kmeans"]["future_coverage"],
            F["semanal"]["c_a"]["wis"],
            F["semanal"]["c_a_vs_poisson"]["wis_gain"],
            F["semanal"]["share_selection"]["chosen_discount"],
            F["alertas"]["detection"]["growth_20"]["cusum"]["detected"],
            F["diario"]["winner"]["wis"],
            F["diario"]["share_selection"]["chosen"]["discount"],
            F["diario"]["composition"]["posterior"]["kappa_p50"],
        ]
        for value in cited:
            self.assertTrue(math.isfinite(float(value)), value)
        if F["composicion"]["kappa"] is not None:
            self.assertGreater(F["composicion"]["kappa"], 100)

    def test_the_daily_winner_is_the_lowest_wis_among_accepted(self):
        counts = self.F["diario"]["counts"]
        accepted = [c for c in counts if c["status"] == "accepted"]
        best = min(accepted, key=lambda c: c["wis"])
        self.assertEqual(self.F["diario"]["winner"]["model_id"], best["model_id"])
        self.assertFalse(self.F["diario"]["change"]["decision"]["adopted"])

    def test_formatters_and_tables(self):
        facts = self.facts
        self.assertEqual(facts.n(1234.5678, 2), "1,234.57")
        self.assertEqual(facts.i(1234.4), "1,234")
        self.assertEqual(facts.pct(0.2964, 0), "30 %")
        self.assertEqual(facts.signed_pct(-0.05), "-5.0 %")
        self.assertEqual(facts.ic({"p025": -0.5, "p975": 0.25}, 2), "[-0.50, +0.25]")
        table = facts.md_table(["a", "b"], [[1, 2]], "lr")
        self.assertIn("|:---|---:|", table)
        self.assertEqual(facts.parse_catalog_value("['x', 'y']"), ["x", "y"])
        self.assertEqual(facts.parse_catalog_value("plain text"), "plain text")


if __name__ == "__main__":
    unittest.main()
