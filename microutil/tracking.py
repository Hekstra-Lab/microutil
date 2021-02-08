__all__ = [
    "construct_cost_matrix",
    "track",
]

import scipy
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import numpy as np


def construct_cost_matrix(prev, curr, weights=[1, 1, 1 / 5]):
    """
    prev : (M, N) array of int
        The previous time point's labels
    curr : (M, N) array of int
        The current time point's labels.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    """
    prev_com = np.asarray(ndi.center_of_mass(prev, prev, np.unique(prev)[1:]))
    curr_com = np.asarray(ndi.center_of_mass(curr, curr, np.unique(curr)[1:]))
    prev_area = np.unique(prev, return_counts=True)[1][1:]
    curr_area = np.unique(curr, return_counts=True)[1][1:]

    prev_features = np.hstack([prev_com, prev_area[:, None]])
    curr_features = np.hstack([curr_com, curr_area[:, None]])

    C = scipy.spatial.distance.cdist(prev_features, curr_features, metric="minkowski", w=weights)
    return C


def track(cells, weights=[1, 1, 1 / 5]):
    """
    Attempt to keep cells' labels the same over time points.

    Parameters
    ----------
    cells : (T, M, N) array-like
        The mask of cell labels. Should be integers.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]

    Returns
    -------
    tracked : (T, M, N) array
        The labels mask with cells tracked through time. So ideally
        a cell labelled 4 at t0 will also be labelled 4 at t1.
    """
    tracked = np.zeros_like(cells)
    tracked[0] = cells[0]

    # for loop for now.
    # xarray rolling operations look promising, but I couldn't get them working.
    # TODO: double and flip the graph to allow for lineage tracking
    # https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf
    for t in range(1, len(cells)):
        C = construct_cost_matrix(tracked[t - 1], cells[t], weights=weights)
        M, N = C.shape
        if M < N:
            # maybe these should be low cost connections?
            C = np.pad(C, ((0, N - M), (0, 0)), constant_values=100000)
        row_ind, col_ind = linear_sum_assignment(C)
        assignments = np.stack([row_ind, col_ind], axis=1)

        for i in range(len(assignments)):
            prev, curr = assignments[i]
            idx = cells[t] == curr + 1
            if i > M:
                tracked[t][idx] = curr + 1
            else:
                tracked[t][idx] = prev + 1
    return tracked
