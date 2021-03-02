__all__ = [
    "apply_unet",
    "threshold_predictions",
    "individualize",
    "point_selector",
    "brush",
    "correct_watershed",
]


# these try except blocks are due to the fact that
# tensorflow doesn't support python 3.9 yet (2021-01-29)
# on macOS big sur napari only works on  python 3.9.0
# So two envs are needed for mac users (e.g. indrani)
# this allows importing this file from either env.
import warnings


import numpy as np
import xarray as xr
import dask.array as da
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_isodata
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed
from .track_utils import _reindex_labels


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
    from ._unet import unet

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
        xarr = xr.DataArray(out, dims=data.dims, coords=data.coords)
        if 'C' in xarr.coords:
            xarr['C'] = 'mask'
        return xarr
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
        The mask is now 0 for background and integers for cell ids
    """

    def _individualize(mask):
        dtr = ndi.morphology.distance_transform_edt(mask)
        topology = -dtr

        peak_idx = peak_local_max(-topology, min_distance)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True

        m_lab = label(peak_mask)

        mask = watershed(topology, m_lab, mask=mask, connectivity=2)
        if min_area is None:
            return mask
        else:
            return _reindex_labels(mask, min_area, inplace=None)[0]

    return xr.apply_ufunc(
        _individualize,
        mask,
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        dask="parallelized",
        vectorize=True,
    )


class point_selector:
    def __init__(self, ax, init_points=None, radius=50, color="red", alpha=0.5):
        """
        Mark points on a plot. left-click to mark a point, and shift-left-click to remove a point.
        The numpy array of points is accessible via the `.points` attribute.

        Parameters
        ----------
        ax : matplotlib axis
        init_points : (N, 2) array-like, optional
        radius : float, default: 50
            The size of the points being marked.
        color : colorlike
            The color of the points
        alpha : float
            The alpha of the points
        """
        self._ax = ax
        self._fig = self._ax.figure
        self.callback_id = self._fig.canvas.mpl_connect("button_press_event", self._callback)
        self._alpha = alpha
        self._radius = radius
        self._color = color
        if init_points is None:
            self._Xs = []
            self._Ys = []
        else:
            init_points = np.asarray(init_points)
            self._Xs = init_points[:, 0].tolist()
            self._Ys = init_points[:, 1].tolist()
        self._scat = ax.scatter(self._Xs, self._Ys, s=radius, color=color, alpha=alpha)

    @property
    def points(self):
        return np.array([self._Xs, self._Ys]).T

    @points.setter
    def points(self, value):
        value = np.asarray(value)
        self._Xs = value[:, 0].tolist()
        self._Ys = value[:, 1].tolist()
        self._scat.set_offsets(value)

    def _callback(self, event):
        if (
            event.button == 1
            and event.xdata is not None
            and event.ydata is not None
            and event.inaxes is self._ax
        ):
            if event.key == "shift":
                idx = np.argmin(
                    cdist([[event.xdata, event.ydata]], np.column_stack([self._Xs, self._Ys]))
                )
                # TODO: add a check for max removal distance?
                self._Xs.pop(idx)
                self._Ys.pop(idx)
            else:
                self._Xs.append(event.xdata)
                self._Ys.append(event.ydata)
            self._scat.set_offsets(np.column_stack([self._Xs, self._Ys]))
            self._fig.canvas.draw_idle()


class brush:
    def __init__(self, ax, arr, erase_value=0, fill_value=1, brush_size=1, alpha=0.5):
        """
        Update a mask by drawing with the mouse. This defaults to erasing mode, but you can
        fill by holding `ctrl` when drawing. This will not modify *arr* inplace, you can get the
        updated version by accessing `brush.arr`.

        Parameters
        ----------
        ax : matplotlib axis
        arr : (X, Y) array-like
            The mask to draw on
        erase_value : number, default: 0
            The value to fill *arr* with when erasing
        fill_value : number, default: 0
            The value to fill *arr* with when erasing
        brush_size : int, default: 1
            The sidelength of the square brush size
        alpha : float, default: 0.5
            The alpha of the mask.
        """
        self._ax = ax
        self._fig = self._ax.get_figure()
        self.arr = np.array(arr)
        self._im = self._ax.imshow(
            self.arr,
            alpha=alpha,
            cmap="Reds",
        )
        self.erase_value = erase_value
        self.fill_value = fill_value
        self.brush_size = brush_size
        self._fig.canvas.mpl_connect("button_press_event", lambda event: self._button(event, True))
        self._fig.canvas.mpl_connect(
            "button_release_event", lambda event: self._button(event, False)
        )
        self._fig.canvas.mpl_connect('axes_leave_event', self._stop)
        self._fig.canvas.mpl_connect("motion_notify_event", self._drag)
        self._active = False

    def _stop(self, event):
        if event.inaxes is self._ax:
            self._active = False

    def _button(self, event, to):
        if (
            event.button == 1
            and event.key in ['', 'ctrl', 'control', None]
            and event.inaxes is self._ax
        ):
            self._active = to

    def _drag(self, event):
        if self._active and event.inaxes is self._ax:
            if event.key in ['ctrl', 'control']:
                val = self.fill_value
            else:
                val = self.erase_value
            # the swap of x and y is intentional to deal with
            # the transpose that imshow introduces
            y = np.int(event.xdata)
            x = np.int(event.ydata)
            if self.brush_size == 1:
                self.arr[x, y] = val
            else:
                delta = self.brush_size / 2
                self.arr[x - delta : x + delta, y - delta : y + delta] = val
            self._im.set_data(self.arr)


def update_ds_seeds(ds, new_seeds, T):
    """
    Replace the watershed_seeds in a dataset for a single time point
    and position. This helper automatically pads when necessary.

    Parameters
    ----------
    new_seeds : (N, 2) array
    """
    if isinstance(new_seeds, np.ndarray):
        new_seeds = xr.DataArray(new_seeds, dims=('points', '_xy'))
    seeds = ds['watershed_seeds']
    new_len = len(new_seeds)
    diff = new_len - ds.dims['points']
    if diff > 0:
        seeds = seeds.pad(pad_width={'points': (0, diff)})
    else:
        new_seeds = new_seeds.pad(pad_width={'points': (0, -diff)})

    seeds[T] = new_seeds

    vs = list(ds.variables.keys())
    vs.remove('watershed_seeds')
    new_ds = ds[vs]
    new_ds['watershed_seeds'] = (ds['watershed_seeds'].dims, seeds)
    return new_ds


def correct_watershed(
    data,
    time_point,
    xlim=None,
    ylim=None,
    point_radius=50,
    point_color='red',
    mask_alpha=0.5,
    mask_cmap='Reds',
    mask_brush_size=1,
):
    """
    Manually correct parts of an image with a bad watershed.
    This will modify the 'watershed_seeds' and 'labels' variables of dat

    Parameters
    ----------
    data : DataSet
        Must have "images", and "mask", "labels", and "watershed_points". And "images"
        must contain "BF"
    time_point : int
    xlim, ylim : tuple of int
    point_radius : float, default: 50
        The size of the points being marked.
    point_color : colorlike
        The color of the points
    point_alpha : float, default: 0.75
        The alpha of the points
    mask_alpha : float, default: 0.5
        The alpha of the mask
    mask_cmap : str of matplotlib cmap, default: 'Reds'
    mask_brush_size : int, default: 1
        The sidelength of the square brush size
    """
    import copy
    import ipywidgets as widgets
    from mpl_interactions import zoom_factory, panhandler
    import matplotlib.pyplot as plt

    def _dims_to_min_max(dims):
        if isinstance(dims, int):
            return (0, dims)
        return np.min(dims), np.max(dims)

    if xlim is None:
        xlim = _dims_to_min_max(data.dims['X'])
    if ylim is None:
        ylim = _dims_to_min_max(data.dims['Y'])

    # create a custom cmap

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under(alpha=0)

    xy_slices = (slice(*xlim), slice(*ylim))
    BF = data['images'].sel(C='BF')[time_point][xy_slices]
    mask = data['mask'][time_point][xy_slices]
    indiv = data['labels'][time_point][xy_slices]
    # keep the max cell number so we don't accidentally create duplicate cells
    # this is used in recalc. Don't constantly call max there to avoid
    # the numbers getting forever larger.
    label_offset = indiv.max().data.item()
    seeds = data['watershed_seeds'][time_point].astype(np.int)
    idx = seeds[:, 0] != np.nan
    idx = np.logical_and(idx, np.logical_and(seeds[:, 0] >= xlim[0], seeds[:, 0] <= xlim[1]))
    idx = np.logical_and(idx, np.logical_and(seeds[:, 1] >= ylim[0], seeds[:, 1] <= ylim[1]))
    subset_seeds = seeds[idx]
    subset_seeds[:, 0] -= xlim[0]
    subset_seeds[:, 1] -= ylim[0]
    subset_seeds = subset_seeds[:, [1, 0]]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 4))
    axs[0].imshow(BF, cmap="gray")
    axs[1].imshow(BF, cmap="gray")
    sel = point_selector(axs[0], init_points=subset_seeds, color=point_color, radius=point_radius)
    new_mask = brush(axs[1], mask, alpha=mask_alpha, brush_size=mask_brush_size)

    out_im = axs[2].imshow(indiv, cmap=cmap)
    uniq = np.sort(np.unique(indiv))
    out_im.norm.vmin = uniq[1]
    out_im.norm.vmax = uniq[-1] + 1

    button = widgets.Button(description='recalc')
    zoom_factory(axs[0])
    zoom_factory(axs[1])
    ph = panhandler(fig, button=3)

    def recalc(_):
        # nonlocal data so that we can reassign the seeds
        nonlocal data
        # putting the ph here hides it away inside the global state of ipywidgets
        # this keeps it from being garbage collected
        # perhaps it's a horrible thing to do...
        # but it is a workaround to being unable to block on a step
        # when using ipywidgets - so I think it's fine.
        ph
        ######
        peak_mask = np.zeros_like(indiv, dtype=bool)
        peak_mask[tuple(sel.points.astype(np.int).T)[::-1]] = True
        topology = -ndi.distance_transform_edt(new_mask.arr)
        indiv2 = watershed(topology, label(peak_mask), mask=new_mask.arr, connectivity=2)
        indiv2[indiv2 != 0] += label_offset
        out_im.set_data(indiv2)
        uniq = np.sort(np.unique(indiv2))
        out_im.norm.vmin = uniq[1]
        out_im.norm.vmax = uniq[-1] + 1

        # update the dataset in place
        data['mask'][time_point][xy_slices] = new_mask.arr
        data['labels'][time_point][xy_slices] = indiv2
        # create new seeds - taking care to include all the ones that
        # are not shown and to avoid any nans or other undesirable values
        # (sometimes that nans ended up as most negative float32)
        new_points = sel.points
        new_seeds = np.zeros_like(sel.points, dtype=np.float32)
        new_seeds[:] = sel.points
        new_seeds[:, 0] += ylim[0]
        new_seeds[:, 1] += xlim[0]
        orig_seeds = seeds[~idx]
        orig_seeds = orig_seeds[orig_seeds[:, 0] != np.nan]
        orig_seeds = orig_seeds[orig_seeds[:, 0] >= 0]
        new_seeds = np.vstack((orig_seeds, new_seeds))

        # do the transpose `[1,0]` to account for the one we did during set up.
        data = update_ds_seeds(data, new_seeds[:, [1, 0]], time_point)

    button.on_click(recalc)
    fig.tight_layout()
    display(button)
