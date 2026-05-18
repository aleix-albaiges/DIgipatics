import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import training_conchv2 as training_conchv2


class NearIdentityLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, training_conchv2.NUM_CLASSES, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[0, 0, 0, 0] = 1.0
            self.proj.weight[1, 1, 0, 0] = 1.0
            self.proj.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        return self.proj(x)


class FixedPatternModel(nn.Module):
    def __init__(self, pattern: torch.Tensor):
        super().__init__()
        self.register_buffer("pattern", pattern)

    def forward(self, x):
        return self.pattern.expand(x.shape[0], -1, -1, -1)


class TestTTAForward(unittest.TestCase):
    def test_d4_tta_matches_single_forward_for_near_identity_model(self):
        model = NearIdentityLogitModel().eval()
        image = torch.arange(1 * 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32) / 255.0

        single = model(image)
        tta_logits = training_conchv2.tta_forward(model, image, scales=(1.0,), use_d4=True)

        self.assertEqual(tuple(tta_logits.shape), (1, training_conchv2.NUM_CLASSES, 32, 32))
        self.assertTrue(torch.allclose(tta_logits, single, atol=1e-3, rtol=1e-3))

    def test_d4_average_is_symmetric_for_fixed_non_symmetric_pattern(self):
        rows = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1)
        cols = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32)
        base = rows * 17.0 + cols * 3.0
        pattern = torch.cat(
            [
                base,
                rows * 5.0 - cols * 2.0,
                torch.sin(base / 11.0),
                torch.cos((rows + 2.0 * cols) / 7.0),
            ],
            dim=1,
        )
        model = FixedPatternModel(pattern).eval()
        image = torch.linspace(0.0, 1.0, steps=3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)

        tta_logits = training_conchv2.tta_forward(model, image, scales=(1.0,), use_d4=True)

        for transform_name in training_conchv2._D4_TRANSFORMS:
            transformed = training_conchv2._apply_d4_transform(tta_logits, transform_name)
            self.assertTrue(
                torch.allclose(tta_logits, transformed, atol=1e-5, rtol=1e-5),
                msg=f"TTA logits are not D4-symmetric for transform {transform_name}",
            )


if __name__ == "__main__":
    unittest.main()
