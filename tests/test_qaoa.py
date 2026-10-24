import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from qaoa_bok.qaoa import (cut_values, expected_best_of_k, noisy_probs, random_regular_graph,
                           statevector_probs, success_prob_best_of_k)


def setup():
    g = random_regular_graph(6, 0)
    return g, cut_values(g)


def test_noisy_matches_statevector_at_zero_noise():
    g, c = setup()
    x = np.array([0.4, 0.3, 0.5, 0.2])
    assert np.allclose(noisy_probs(g, c, x, 0.0, 0.0), statevector_probs(c, 6, x), atol=1e-10)


def test_noise_normalised_and_tends_to_uniform():
    g, c = setup()
    x = np.array([0.4, 0.3])
    p = noisy_probs(g, c, x, 0.05, 0.005)
    assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    assert abs(p @ c - c.mean()) < abs(statevector_probs(c, 6, x) @ c - c.mean())


def test_best_of_k_monotone_and_bounded():
    g, c = setup()
    u = np.full(len(c), 1 / len(c))
    a, b = expected_best_of_k(u, c, 1), expected_best_of_k(u, c, 50)
    assert abs(a - c.mean()) < 1e-9 and a < b <= c.max()
    assert 0 < success_prob_best_of_k(u, c, 5) < 1
