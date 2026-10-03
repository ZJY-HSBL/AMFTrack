import numpy as np
from amftrack.matching.ciou import ciou_loss


def test_ciou_identical_is_zero():
    box = np.array([0, 0, 10, 20], dtype=float)
    assert abs(ciou_loss(box, box)) < 1e-9


def test_ciou_non_overlap_is_positive():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([20, 20, 30, 30], dtype=float)
    assert ciou_loss(a, b) > 1.0
