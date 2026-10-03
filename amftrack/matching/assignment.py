from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    safe = cost.copy()
    finite = np.isfinite(safe)
    if not finite.any():
        return [], list(range(n_rows)), list(range(n_cols))
    safe[~finite] = threshold + 1e6

    rows, cols = linear_sum_assignment(safe)
    matches: list[tuple[int, int]] = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        if np.isfinite(cost[r, c]) and cost[r, c] <= threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
