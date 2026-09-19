"""Guard the Paper 4 chart against restoring a superseded partial endpoint."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Paper4SpectralAnchorTests(unittest.TestCase):
    def test_h_ch4_plot_uses_anchored_final_checkpoint(self):
        source = ROOT / "scripts/generate_paper4_figures.py"
        assignments = {}
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {"ns", "fiedlers"}):
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
        anchor = json.loads((ROOT / "results/attractor-hermes-anchor/H-ch4/results.json")
                            .read_text(encoding="utf-8"))
        final = anchor["checkpoints"][-1]
        self.assertEqual(final["step"], 10000)
        plotted = assignments["fiedlers"][assignments["ns"].index(4)]
        self.assertEqual(plotted, round(final["fiedler_mean"], 4))


if __name__ == "__main__":
    unittest.main()
