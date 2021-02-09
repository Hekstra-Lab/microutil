__all__ = [
    "reindex_labels",
    "reindex_labels_return",
    "check_cell_numbers",
    "find_duplicate_labels",
]
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt


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


def check_cell_numbers(BF, mask, check_after=True, correct=True, bad_frames=None, min_area=25):
    """
    check if the number of cells ever decreases from one frame to the next. If there is a decrease
    and *correct* is True then open napari to allow for manual correction.
    WARNING: currently only allows for a time dimesion. Will need updating for more complex indexing

    Parameters
    ----------
    BF : (..., X, Y) array
        The BF images to use as background
    mask : (..., X, Y) array
    check_after : bool, default: True
        Whether to run the checking code after the manual updates.
    correct : bool, default: True
        Whether to open Napari to correct the bad frames.
    bad_frames : list, optional
        If not None then this will be used instead of the initial check.
        This is to enable calling this function in a loop and not doubling up on checks.
    min_area : number or None, default: 25

    Returns
    -------
    bad_frames : list, or None
        If *check_after* is True then this will be frames after

    Examples
    --------
    >>> bad_frames = check_cell_numbers(BF, tracked)
    >>> while bad_frames is not None:
        print("========")
        print(bad_frames)
        bad_frames = check_cell_numbers(BF, tracked)

    """
    if mask.ndim != 3:
        raise ValueError(
            "Not yet supported - If you run into this error"
            "then it is time to start supporting this"
            "tell Ian to get it together and make it happen..."
        )

    def _check(arr):
        for i in range(len(arr)):
            mask[i] = mu.track_utils.reindex_labels(arr[i], min_area=min_area)
        bad_frames = []
        for i in range(1, len(mask)):
            prev_N = len(np.unique(mask[i - 1]))
            curr_N = len(np.unique(mask[i]))
            if prev_N > curr_N:
                print(i, prev_N, curr_N)
                bad_frames.append(i)
        return bad_frames

    def _empty_check(l):
        if len(l) == 0:
            return None

    if bad_frames is None:
        bad_frames = _check(mask)

    if correct:
        for i in bad_frames:
            fixed = mu.manual_segmentation(BF[i - 1 : i + 1], mask[i - 1 : i + 1])
            mask[i - 1 : i + 1] = fixed
    else:
        return _empty_check(bad_frames)

    if check_after:
        return _empty_check(_check(mask))
    return None


def find_duplicate_labels(frame):
    """
    Open a plot with a slider to tanually search a frame
    in order to check if any two cells share a label.

    Parameters
    ----------
    frame : (X, Y) array-like of int
        The labels of the frame.
    """
    slider = widgets.IntSlider(min=0, max=np.max(np.unique(frame)))
    fig = plt.figure()
    im = plt.imshow(frame == 0)

    def update(change):
        im.set_data(frame == change['new'])
        fig.canvas.draw()

    slider.observe(update, names='value')
    display(slider)
