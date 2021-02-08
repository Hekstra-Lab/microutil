__all__ = [
    "apply_unet",
    "unet",
    "threshold_predictions",
    "individualize",
    "manual_segmentation",
]


# these try except blocks are due to the fact that
# tensorflow doesn't support python 3.9 yet (2021-01-29)
# on macOS big sur napari only works on  python 3.9.0
# So two envs are needed for mac users (e.g. indrani)
# this allows importing this file from either env.
import warnings

try:
    from ._unet import unet
except ImportError:
    unet = None
    warnings.warn(
        "Could not import our unet model. You likely do not have"
        "Tensorflow installed. The function apply_unet will fail if you call it",
        stacklevel=3,
    )
try:
    import napari
except ImportError:
    napari = None
    warnings.warn(
        "Could not import napari. The function *manual_segmentation* will fail if you call it",
        stacklevel=3,
    )

import numpy as np
import xarray as xr
import dask.array as da
import scipy.ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_isodata
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed


def apply_unet(data, model):
    """
    Apply the UNET to make pixel-wise predictions of cell vs not-cell

    Parameters
    ----------
    data : array-like
        The final two axes should be XY
    model : str or unet instance
        Either an instance of a loaded unet model or a path the model weights

    Returns
    -------
    mask : array-like
        The predicted mask
    """
    if unet is None:
        raise ImportError("You must install Tensorflow in order to use this function.")
    is_xarr = False
    if isinstance(data, xr.DataArray):
        arr = data.values
        is_xarr = True
    else:
        arr = data

    # TODO make this xarray/dask parallelize if applicable
    arr = np.vectorize(equalize_adapthist, signature="(x,y)->(x,y)")(arr)

    orig_shape = arr.shape
    row_add = 16 - orig_shape[-2] % 16
    col_add = 16 - orig_shape[-1] % 16
    npad = [(0, 0)] * arr.ndim
    npad[-2] = (0, row_add)
    npad[-1] = (0, col_add)

    arr = np.pad(arr, npad)

    # manipulate into shape that tensorflow expects
    if len(orig_shape) == 2:
        arr = arr[None, :, :, None]
    else:
        arr = arr.reshape(
            (
                np.prod(orig_shape[:-2]),
                orig_shape[-2] + row_add,
                orig_shape[-1] + col_add,
                1,
            )
        )

    if isinstance(model, str):
        model = unet(model, (None, None, 1))

    # need the final reshape to squeeze off a potential leading 1 in the shape
    # but we can't squeeze because that might remove axis with size 1
    out = model.predict(arr)[..., :-row_add, :-col_add, 0].reshape(orig_shape)
    if is_xarr:
        return xr.DataArray(out, dims=data.dims, coords=data.coords)
    else:
        return out


def threshold_predictions(predictions, threshold=None):
    """
    Parameters
    ----------
    predictions : array-like
    threshold : float or None, default: None
        If None the threshold will automatically determined using
        skimage.filters.threshold_isodata

    Returns
    -------
    mask : array of bool
    """
    if threshold is None:
        threshold = threshold_isodata(np.asarray(predictions))
    return predictions > threshold


def individualize(mask, min_distance=10, connectivity=2, min_area=25):
    """
    Turn a boolean mask into a a mask of cell ids

    Parameters
    ---------
    mask : array-like
        Last two dimensions should be XY
    min_distance : int, default: 10
        Passed through to scipy.ndimage.morphology.distance_transform_edt
    connectivity : int, default: 2
        Passed through to skimage.segmentation.watershed
    min_area : number, default: 25
        The minimum number of pixels for an object to be considered a cell.
        If *None* then no cuttoff will be applied, which can reduce computation time.

    Returns
    -------
    cell_ids : array-like of int
        The mask is now 0 for backgroud and integers for cell ids
    """

    def _cleanup(frame):
        out = np.zeros_like(frame)
        next_cell_id = 1
        # TODO: figure out how to return these areas as well
        # seems tricky, probably won't.... - Ian 2021-02-08
        ids, areas = np.unique(frame, return_counts = True)
        areas[1:] > min_area
        for i, area in zip(ids[1:], areas[1:]):
            if area > min_area:
                idx = frame == i
                out[idx] = next_cell_id
                next_cell_id += 1
        return out

    def _individualize(mask):
        dtr = ndi.morphology.distance_transform_edt(mask)
        topology = -dtr

        peak_idx = peak_local_max(-topology, min_distance)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True

        m_lab = label(peak_mask)

        mask =  watershed(topology, m_lab, mask=mask, connectivity=2)
        if min_area is None:
            return mask
        else:
            return _cleanup(mask)


    return xr.apply_ufunc(
        _individualize,
        mask,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
    )


def manual_segmentation(img, mask=None):
    """
    Open up Napari set up for manual segmentation. Adds these custom keybindings:
    q : erase
    w : fill
    e : paint
    r : pick
    t : create new label

    also sets it so scrolling in paint mode will modifying brush size.

    Parameters
    ----------
    img : image-like
        Last to dims should be XY. You probably want this to be a BF image.
    mask : array-like or None
        If array-like it should be broadcastable to the same dims as *img*

    Returns
    -------
    mask :
        The mask that was updated by user interactions
    """
    if napari is None:
        raise ImportError("You must install Napari in order to use this function.")
    if mask is None:
        # needs to be numpy as all other options do not seem to work
        # see https://github.com/napari/napari/issues/2190
        mask = np.zeros_like(img, dtype=np.int)
        # if isinstance(img, np.ndarray):
        #     mask = np.zeros_like(img,dtype=np.int)
        # elif isinstance(img, xr.DataArray):
        #     mask = nr.zeros_like(img, dtype=np.int)
        # elif isinstance(img, da.Array):
        #     mask = np.zeros_like(img)
    elif not isinstance(mask, np.ndarray):
        print("casting mask to numpy array")
        print("see https://github.com/napari/napari/issues/2190 for details")
        mask = np.array(mask)

    with napari.gui_qt():
        # create the viewer and add the cells image
        viewer = napari.view_image(img, name="cells")
        # add the labels
        labels = viewer.add_labels(mask, name="segmentation")
        # Add more keybinds for better ergonomics
        @labels.bind_key("q")
        def paint_mode(viewer):
            labels.mode = "erase"

        @labels.bind_key("w")
        def paint_mode(viewer):
            labels.mode = "fill"

        @labels.bind_key("e")
        def paint_mode(viewer):
            labels.mode = "paint"

        @labels.bind_key("r")
        def paint_mode(viewer):
            labels.mode = "pick"

        @labels.bind_key("t")
        def new_cell(viewer):
            labels.selected_label = labels.data.max() + 1

        # scrolling in paint mode changes the brush size
        def brush_size_callback(layer, event):
            if labels.mode in ["paint", "erase"]:
                if event.delta[1] > 0:
                    labels.brush_size += 1
                else:
                    labels.brush_size = max(labels.brush_size - 1, 1)

        labels.mouse_wheel_callbacks.append(brush_size_callback)

    if isinstance(img, xr.DataArray):
        return xr.DataArray(labels.data, coords=img.coords, dims=img.dims)
    else:
        return labels.data
