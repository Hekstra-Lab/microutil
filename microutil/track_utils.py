__all__ = [
    "reindex_labels",
    "reindex_labels_return",
]
import numpy as np


def _reindex_labels(ids, areas, frame, min_area):
    # TODO: make this less for loopy :(
    out = np.zeros_like(frame)
    new_ids = [0]
    next_cell_id = 1
    new_areas = [areas[0]]
    if min_area is None:
        min_area = np.min(areas)
    for i, area in zip(ids[1:], areas[1:]):
        if area >= min_area:
            new_areas.append(area)
            new_ids.append(next_cell_id)
            idx = frame == i
            out[idx] = next_cell_id
            next_cell_id += 1
    return out, np.array(new_ids), np.array(new_areas)


def reindex_labels_return(frame, min_area=None):
    """
    Reindex the cell labels and return the updated frame as well
    as the new ids and new area.

    Parameters
    ----------
    frame : (X, Y) array-like
    min_area : number or None

    Returns
    -------
    updated_frame : (X, Y) array
        The frame with small patches removed and the labels do not skip any integers.
    ids : 1d array of int
        The unique labels in the updated frame
    areas : 1d array
        The number of pixels associated with each of the labels in the updated frame
    """
    ids, areas = np.unique(frame, return_counts=True)
    if np.all(np.arange(ids.max() + 1) != ids) or (
        min_area is not None and np.any(areas < min_area)
    ):
        return _reindex_labels(ids, areas, frame, min_area)
    else:
        return frame, ids, areas


def reindex_labels(frame, min_area=None):
    """
    Reindex the cell labels and only return the updated frame.

    Parameters
    ----------
    frame : (X, Y) array-like
    min_area : number or None

    Returns
    -------
    updated_frame : (X, Y) array
        The frame with small patches removed and the labels do not skip any integers.
    """
    return reindex_labels_return(frame, min_area)[0]
