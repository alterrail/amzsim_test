import unittest
from pathlib import Path

import numpy as np

from simulate import load_cones, middle_line, pose_at, scan


class SimulationTests(unittest.TestCase):
    def test_fov_range_and_heading(self):
        points = np.array([[5, 0], [-5, 0], [0, 5], [31, 0], [5, 5]])
        visible, detected, probability = scan(points, np.zeros(2), 0, 30, 20, np.random.default_rng(1))
        np.testing.assert_array_equal(visible, [True, False, False, False, True])
        self.assertTrue(np.all(~detected | visible))
        self.assertTrue(np.all(probability[~visible] == 0))
        visible, _, _ = scan(points, np.zeros(2), np.pi / 2, 30, 20, np.random.default_rng(1))
        np.testing.assert_array_equal(visible, [False, False, True, False, True])

    def test_probability_decreases_and_matches_sampled_frequency(self):
        points = np.array([[1, 0], [10, 0], [25, 0]])
        rng = np.random.default_rng(19)
        hits = np.zeros(3)
        for _ in range(10000):
            _, detected, probability = scan(points, np.zeros(2), 0, 30, 20, rng)
            hits += detected
        self.assertTrue(np.all(np.diff(probability) < 0))
        np.testing.assert_allclose(hits / 10000, probability, atol=0.02)

    def test_open_straight_middle_line_and_stop(self):
        points = np.array([[0, 2], [10, 2], [10, -2], [0, -2]])
        line = middle_line(points, np.array(['blue', 'blue', 'yellow', 'yellow']), False)
        np.testing.assert_allclose(line[:, 1], 0)
        position, heading = pose_at(line, 4, False)
        np.testing.assert_allclose(position, [4, 0])
        self.assertAlmostEqual(heading, 0)
        np.testing.assert_allclose(pose_at(line, 100, False)[0], [10, 0])

    def test_closed_track_wraps(self):
        points, tags = load_cones(Path(__file__).with_name('sample_track.csv'))
        line = middle_line(points, tags)
        length = np.linalg.norm(np.diff(np.vstack([line, line[0]]), axis=0), axis=1).sum()
        np.testing.assert_allclose(pose_at(line, 3, True)[0], pose_at(line, length + 3, True)[0], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
