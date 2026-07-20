import unittest
from types import SimpleNamespace

try:
    import torch
    from kvcachebench.presses import attach_retained_tracker
except ImportError:
    torch = None
    attach_retained_tracker = None


class PyramidKVPress:
    compression_ratio = 0.5

    def __init__(self):
        self.budget_calls = []

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return torch.arange(keys.shape[2], dtype=keys.dtype).view(1, 1, -1)

    def get_layer_budget(self, module, k_len):
        self.budget_calls.append((module.layer_idx, k_len))
        return 3


@unittest.skipIf(
    torch is None or attach_retained_tracker is None,
    "torch is required for tracker tests",
)
class PyramidTrackerTest(unittest.TestCase):
    def test_preserves_layer_adaptive_budget(self):
        press = PyramidKVPress()
        tracker = attach_retained_tracker(press)
        module = SimpleNamespace(layer_idx=7, head_dim=2)
        keys = torch.arange(20, dtype=torch.float32).view(1, 1, 10, 2)
        values = keys + 100

        compressed_keys, compressed_values = press.compress(
            module, None, keys, values, None, {}
        )

        self.assertEqual(press.budget_calls, [(7, 10)])
        self.assertEqual(compressed_keys.shape[2], 3)
        self.assertEqual(compressed_values.shape[2], 3)
        self.assertEqual(tracker.records["7"], [[9, 8, 7]])


if __name__ == "__main__":
    unittest.main()
