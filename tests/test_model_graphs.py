import unittest

import pymc as pm

from src.models.model_graphs import model_graph


class ModelGraphTests(unittest.TestCase):
    def test_graph_has_the_title_and_every_variable(self):
        with pm.Model(coords={"cluster": range(3)}) as model:
            log_kappa = pm.Normal("log_kappa", 0, 1)
            kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))
            pm.Normal("observed", kappa, 1, observed=[1.0, 2.0, 3.0], dims="cluster")

        source = model_graph(model, "Demo model").source

        self.assertIn('label="Demo model"', source)
        for name in ("log_kappa", "kappa", "observed", "cluster (3)"):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
