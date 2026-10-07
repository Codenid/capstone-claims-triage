import json
from pathlib import Path
import unittest

import yaml

from src.data.params import PROJECT_ROOT, load_params, preparation_settings

STAGES = (
    "type_data",
    "normalize_text",
    "apply_taxonomy",
    "build_targets",
    "finalize_prepared",
)
CONTRACT = PROJECT_ROOT / "reports/modeling/input_contract.json"


class PreparationParamsTests(unittest.TestCase):
    def test_every_stage_declares_its_section_in_dvc_yaml(self):
        stages = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        stages = stages["stages"]

        for stage in STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(stages[stage]["params"], [f"preparacion.{stage}"])
                self.assertIn("src/data/params.py", stages[stage]["deps"])
                settings = preparation_settings(stage)
                self.assertEqual(stages[stage]["outs"][0], settings["salida"])
                self.assertIn(settings["entrada"], stages[stage]["deps"])

    def test_params_match_the_frozen_input_contract(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        versions = contract["versions"]
        normalizer = preparation_settings("normalize_text")
        taxonomy = preparation_settings("apply_taxonomy")
        targets = preparation_settings("build_targets")
        final = preparation_settings("finalize_prepared")

        self.assertEqual(normalizer["version"], versions["text_normalizer"])
        self.assertEqual(taxonomy["version"], versions["taxonomy_version"])
        self.assertEqual(taxonomy["estado"], versions["taxonomy_status"])
        freeze = taxonomy["fecha_congelamiento"].isoformat()
        self.assertEqual(freeze, versions["taxonomy_freeze_date"])
        self.assertEqual(targets["version"], versions["targets_periods_version"])
        periods = set(targets["inicio_de_periodo"]) | {"context_2015_2022"}
        self.assertEqual(periods, set(contract["period_rows"]))
        self.assertEqual(final["filas_esperadas"], contract["rows"])

    def test_unknown_stage_is_an_error(self):
        with self.assertRaises(KeyError):
            preparation_settings("no_such_stage")
        self.assertIsInstance(load_params(Path(PROJECT_ROOT / "params.yaml")), dict)


if __name__ == "__main__":
    unittest.main()
