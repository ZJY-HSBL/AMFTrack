import numpy as np
from amftrack.matching.assignment import linear_assignment


def test_linear_assignment_threshold():
    cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=float)
    matches, ur, uc = linear_assignment(cost, 0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert ur == []
    assert uc == []
