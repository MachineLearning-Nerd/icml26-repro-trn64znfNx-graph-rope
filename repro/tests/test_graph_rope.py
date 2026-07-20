import unittest

from repro.src.verify_graph_rope import (
    claim_effective_resistance,
    claim_grid_rope,
    claim_linear_attention,
    path_mode,
    spectral_path_resistance,
)


class GraphRoPETest(unittest.TestCase):
    def test_fiedler_coordinate_is_strictly_monotone(self):
        coordinate = [path_mode(23, 1, i) for i in range(23)]
        self.assertTrue(all(a > b for a, b in zip(coordinate, coordinate[1:])))

    def test_path_resistance_is_electrical_distance(self):
        self.assertAlmostEqual(spectral_path_resistance(31, 4, 27), 23.0, places=10)

    def test_all_claim_certificates(self):
        self.assertEqual(claim_grid_rope()["outcome"], "passed")
        self.assertEqual(claim_effective_resistance()["outcome"], "passed")
        self.assertEqual(claim_linear_attention()["outcome"], "passed")


if __name__ == "__main__":
    unittest.main()
