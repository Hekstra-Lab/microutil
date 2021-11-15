__all__ = [
    "reindex",
    "reindex_labels",
    "check_cell_numbers",
    "find_duplicate_labels",
    "find_bad_frames",
    "clipped_viridis",
]
from copy import copy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage.segmentation import relabel_sequential

from ._names import POS, TIME
from .array_utils import not_xr, zeros_like

# clipped_virids is a useful colormap for looking at
# individually labelled cell masks in matplotlib.
clipped_viridis = copy(plt.cm.viridis)
clipped_viridis.set_under(alpha=0)


def _reindex_labels(arr, min_area, inplace=None):
    ids, areas = np.unique(arr, return_counts=True)
    if not np.array_equal(np.arange(ids.max() + 1), ids) or (
        min_area is not None and np.any(areas < min_area)
    ):
        # TODO: make this less for loopy :(
        # or just make it very cython
        out = np.zeros_like(arr)
        new_ids = [0]
        next_cell_id = 1
        new_areas = [areas[0]]
        if min_area is None:
            min_area = np.min(areas)
        for i, area in zip(ids[1:], areas[1:]):
            if area >= min_area:
                new_areas.append(area)
                new_ids.append(next_cell_id)
                idx = not_xr(arr == i)
                out[idx] = next_cell_id
                next_cell_id += 1
        if inplace is not None:
            inplace[:] = out
        return out, np.array(new_ids), np.array(new_areas)
    elif inplace is not None:
        inplace[:] = arr
    return arr, ids, areas


def _process_dims(dims, names):
    """
    Check if an xarray's dims contain something that
    is in *names*

    Parameters
    ----------
    dims : tuple of str
    names : str or container of str

    Returns
    -------
    i : int
        The index in dims
    """
    if isinstance(names, int):
        return names
    for i, d in enumerate(dims):
        if d.lower() in names:
            return i
    return None


def reindex(arr, min_area=None, inplace=True, time_axis="infer", pos_axis="infer"):
    """
    Reindex the cell labels and return the updated array.
    See reindex_multi

    Parameters
    ----------
    arr : (T, X, Y) or (X, Y) array-like
        The mask to reindex
    inplace : bool, default: True
    time_axis : int or str, optional
        Which axis to treat as the time axis. The default will
        detect standard labels for xarray. Set to None for no time axis.
    pos_axis : int or str, optional
        Which axis to treat as the position axis. The default will
        detect standard labels for xarray. If *arr* is a numpy array with
        ndim 4 then pos_axis will be assumed to be 1. Set to None to force no
        inference and no looping
    Returns
    -------
    modified : array-like
        The modified array. This is returned even if *inplace* is True.
    """
    # start: nonsense code to infer axis position
    if isinstance(time_axis, str):
        if time_axis == "infer":
            time_axis = TIME
        if isinstance(arr, xr.DataArray):
            time_axis = _process_dims(arr.dims, time_axis)
        elif isinstance(arr, np.ndarray):
            time_axis = 0
        else:
            time_axis = None

    if arr.ndim > 3 and isinstance(pos_axis, str):
        if pos_axis == "infer":
            pos_axis = POS
        if isinstance(arr, xr.DataArray):
            pos_axis = _process_dims(arr.dims, pos_axis)
        elif isinstance(arr, np.ndarray):
            pos_axis = 1
        else:
            pos_axis = None
    else:
        pos_axis = None

    if arr.ndim == 2:
        time_axis = None
        pos_axis = None
    # end: nonsense code to infer axis position

    if not inplace:
        modified = zeros_like(arr)
    else:
        modified = arr[:]

    def _process_times(frames, modified):
        if time_axis is not None:
            N_time = arr.shape[time_axis]
            for t in range(N_time):
                # TODO: allow reindex to take the ids over all time
                # but the areas by time point
                # idk if that's actually helpful.
                # it may actually be harmful in the case of a cell leaving
                _reindex_labels(frames[t], min_area, modified[t])
        else:
            _reindex_labels(arr, min_area, modified[:])

    if pos_axis is not None:
        for p in range(arr.shape[pos_axis]):
            _process_times(arr[:, p], modified[:, p])
    else:
        _process_times(arr, modified[:])
    return modified


def check_cell_numbers(BF, mask, check_after=True, correct=True, bad_frames=None, min_area=25):
    """
    check if the number of cells ever decreases from one frame to the next. If there is a decrease
    and *correct* is True then open napari to allow for manual correction.
    WARNING: currently only allows for a time dimesion. Will need updating for more complex indexing
    This requires that time is the first dimension and XY as the final two dimensions.

    Parameters
    ----------
    BF : (T, ..., X, Y) array
        The BF images to use as background.
    mask : (T, ..., X, Y) array
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
    from .napari_wrappers import manual_segmentation

    time_axis = 0
    N_time = mask.shape[time_axis]

    def _check(arr):
        for i in np.ndindex(arr.shape[:-2]):
            mask[i] = reindex(arr[i], min_area=min_area)
        bad_frames = []
        for i in range(1, N_time):
            prev = arr[i - 1]
            curr = arr[i]
            for j in np.ndindex(prev.shape[:-2]):
                prev_N = len(np.unique(prev[j]))
                curr_N = len(np.unique(curr[j]))
                if prev_N > curr_N:
                    print(i, j, prev_N, curr_N)
                    bad_frames.append((i, j))
        return bad_frames

    def _empty_check(l):  # noqa E741
        if len(l) == 0:
            return None
        else:
            return l

    if bad_frames is None:
        bad_frames = _check(mask)

    if correct:
        for i, j in bad_frames:
            if len(j) > 0:
                frame_BF = BF[[i - 1, i]][:, list(j)]
                # frame_BF = BF[[i - 1, i], list(j)]
                frame_mask = mask[[i - 1, i], list(j)]
            else:
                # there must be a better way...
                # this is to defend against j = []
                # which brings different errors for xarray and numpy
                # and both errors are moderately confusing.
                frame_BF = BF[[i - 1, i]]
                frame_mask = mask[[i - 1, i]]

            fixed = manual_segmentation(frame_BF, frame_mask)
            if len(j) > 0:
                mask[[i - 1, i], list(j)] = fixed
            else:
                mask[[i - 1, i]] = fixed

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
    try:
        from IPython.display import display

        display(slider)
    except ImportError:
        pass


def find_bad_frames(ds, reindex=True):
    """
    Find the frames that have fewer cells than the previous frame.

    Parameters
    ----------
    ds : (S, T, ..., Y, X) DataSet
    reindex : bool, default: True
        Whether to reindex each frame as well.

    Returns
    -------
    bad : list
        With entries (s, t) of the frames that have fewer cells than the previous
        frame.
    """
    bad = []
    for s in range(ds.dims["S"]):
        if reindex:
            ds["labels"][s, 0] = relabel_sequential(ds['labels'][s, 0].values)[0]
        for t in range(1, ds.dims["T"]):
            if reindex:
                ds["labels"][s, t] = relabel_sequential(ds['labels'][s, t].values)[0]
            prev_max = np.max(ds["labels"][s][t - 1].values)
            curr_max = np.max(ds["labels"][s][t].values)
            if curr_max < prev_max:
                print(f"{s=}, {t=} - {prev_max} {curr_max}")
                bad.append((s, t))
    return bad

# def find_bad_tracking(ds, max_distance=50, label_name='labels', dims='STCZYX'):
# 
#     coms = center_of_mass(ds, label_name=label_name, dims=dims)


def reindex_labels(ds):
    """
    Make the cell labels sequential. Modifies the *labels* variable in place.

    Parameters
    ----------
    ds : (S, T, ..., Y, X) Dataset
    """
    for s in range(ds.dims['S']):
        for t in range(ds.dims['T']):
            ds['labels'][s, t] = relabel_sequential(ds['labels'][s, t].values)[0]
