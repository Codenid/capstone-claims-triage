import unittest

import networkx as nx

from src.models.model_graphs import layered_positions, node_label


class ModelGraphTests(unittest.TestCase):
    def test_inputs_sit_above_what_they_feed_and_summaries_below_their_source(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["log_kappa", "observed"], shape="ellipse")
        boxes = ["kappa", "rho", "concentration", "share", "total"]
        graph.add_nodes_from(boxes, shape="box")
        graph.add_edges_from(
            [
                ("log_kappa", "kappa"),
                ("kappa", "rho"),
                ("kappa", "concentration"),
                ("share", "concentration"),
                ("concentration", "observed"),
                ("total", "observed"),
            ]
        )

        rows = {node: row for node, (_, row) in layered_positions(graph).items()}

        self.assertEqual(rows["total"], rows["observed"] + 1)
        self.assertEqual(rows["share"], rows["concentration"] + 1)
        self.assertEqual(rows["rho"], rows["kappa"] - 1)
        self.assertGreater(rows["log_kappa"], rows["kappa"])

    def test_label_shows_name_distribution_and_dimensions(self):
        attributes = {
            "label": "observed\n~\nDirichlet_multinomial",
            "cluster": "clusterweek (87) x cluster (40)",
        }

        self.assertEqual(
            node_label("observed", attributes),
            "observed\n~ Dirichlet_multinomial\nweek (87) x cluster (40)",
        )


if __name__ == "__main__":
    unittest.main()
