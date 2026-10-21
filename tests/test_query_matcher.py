"""Analytic warp controls; no checkpoints, GPU or network required."""

import unittest

import torch

from romatch.query_matcher import TinyRoMaQueryMatcher, sample_query_points


def field(source_size, target_size, offset=(0.0, 0.0), field_size=None):
    h, w = field_size or source_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    norm = torch.stack((2 * (x + 0.5) / w - 1, 2 * (y + 0.5) / h - 1), -1)
    src_px = (norm + 1) * torch.tensor(source_size[::-1]) / 2 - 0.5
    dst = (
        2 * (src_px + torch.tensor(offset) + 0.5) / torch.tensor(target_size[::-1]) - 1
    )
    return torch.cat((norm, dst), -1), torch.ones(h, w)


class FakeTiny:
    device = torch.device("cpu")

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def match(self, a, b, *, batched):
        assert batched is False
        self.calls.append((a, b))
        return self.outputs[len(self.calls) - 1]


class QueryMatcherTests(unittest.TestCase):
    def test_identity_subpixel_boundaries_and_order(self):
        q = torch.tensor([[79.0, 47.0], [0.0, 0.0], [20.25, 13.75], [20.25, 13.75]])
        w, c = field((48, 80), (48, 80))
        r = sample_query_points(w, c, q, (48, 80), (48, 80))
        torch.testing.assert_close(r.right_points, q, atol=1e-5, rtol=0)
        self.assertTrue(r.valid.all())

    def test_translation_unequal_sizes_and_lower_resolution_field(self):
        q = torch.tensor([[30.25, 20.75], [10.5, 10.5]])
        w, c = field((64, 96), (80, 128), (5.5, -2.25), (16, 24))
        r = sample_query_points(w, c, q, (64, 96), (80, 128))
        torch.testing.assert_close(
            r.right_points, q + torch.tensor([5.5, -2.25]), atol=2e-5, rtol=0
        )
        self.assertTrue(r.valid.all())

    def test_invalid_queries_and_target_bounds(self):
        q = torch.tensor([[-1.0, 10.0], [float("nan"), 2.0], [50.0, 20.0], [0.0, 10.0]])
        w, c = field((64, 64), (64, 64), (-5.0, 0.0))
        r = sample_query_points(w, c, q, (64, 64), (64, 64))
        self.assertEqual(r.valid.tolist(), [False, False, True, False])

    def test_confidence_is_bilinearly_interpolated(self):
        w, c = field((64, 64), (64, 64))
        c[:, 11:] = 0
        r = sample_query_points(w, c, torch.tensor([[10.25, 15.5]]), (64, 64), (64, 64))
        torch.testing.assert_close(r.confidence, torch.tensor([0.75]))

    def test_wrapper_masks_and_stable_output(self):
        a, b = torch.zeros(1, 3, 64, 96), torch.ones(1, 3, 80, 128)
        q = torch.tensor([[20.25, 20.75], [40.0, 30.0]])
        fake = FakeTiny(
            [
                field((64, 96), (80, 128), (5.0, 0.0)),
            ]
        )
        mask = torch.ones(80, 128, dtype=torch.bool)
        mask[30, 45] = False
        r = TinyRoMaQueryMatcher(fake).match(a, b, q, right_mask=mask)
        self.assertEqual(r.valid.tolist(), [True, False])
        self.assertEqual(r.right_points.shape, q.shape)
        torch.testing.assert_close(
            r.right_points, q + torch.tensor([5.0, 0.0]), atol=1e-5, rtol=0
        )
        self.assertEqual(len(fake.calls), 1)
        self.assertIs(fake.calls[0][0], a)
        self.assertIs(fake.calls[0][1], b)

    def test_wrapper_rejects_low_confidence(self):
        a = torch.zeros(1, 3, 64, 64)
        q = torch.tensor([[20.0, 20.0], [40.0, 30.0]])
        w, c = field((64, 64), (64, 64))
        c[30, 40] = 0.1
        fake = FakeTiny([(w, c)])
        r = TinyRoMaQueryMatcher(fake).match(a, a, q)
        self.assertEqual(r.valid.tolist(), [True, False])
        self.assertEqual(len(fake.calls), 1)

    def test_forward_only_and_empty_queries(self):
        a = torch.zeros(1, 3, 64, 64)
        fake = FakeTiny([field((64, 64), (64, 64))])
        matcher = TinyRoMaQueryMatcher(fake)
        r = matcher.match(a, a, torch.tensor([[20.0, 20.0]]))
        self.assertTrue(r.valid.all())
        self.assertEqual(len(fake.calls), 1)
        r = matcher.match(a, a, torch.empty(0, 2))
        self.assertEqual(r.right_points.shape, (0, 2))
        self.assertEqual(len(fake.calls), 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_sampling_matches_cpu_at_edges_and_subpixels(self):
        w, c = field((64, 96), (80, 128), (5.5, 2.25))
        q = torch.tensor([[0.0, 0.0], [95.0, 63.0], [12.25, 20.75], [-1.0, 2.0]])
        cpu = sample_query_points(w, c, q, (64, 96), (80, 128))
        gpu = sample_query_points(
            w.to("mps"), c.to("mps"), q.to("mps"), (64, 96), (80, 128)
        )
        torch.testing.assert_close(
            gpu.right_points.cpu(), cpu.right_points, atol=3e-5, rtol=0
        )
        torch.testing.assert_close(gpu.confidence.cpu(), cpu.confidence)
        torch.testing.assert_close(gpu.valid.cpu(), cpu.valid)

    def test_bad_inputs(self):
        fake = FakeTiny([])
        for threshold in [-1.0, 2.0, float("nan")]:
            with self.assertRaises(ValueError):
                TinyRoMaQueryMatcher(fake, confidence_threshold=threshold)
        matcher = TinyRoMaQueryMatcher(fake)
        a = torch.zeros(1, 3, 64, 64)
        with self.assertRaises(ValueError):
            matcher.match(a, a, torch.tensor([[2, 3]]))
        with self.assertRaises(ValueError):
            matcher.match(
                a, a, torch.tensor([[2.0, 3.0]]), left_mask=torch.ones(64, 64)
            )
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
