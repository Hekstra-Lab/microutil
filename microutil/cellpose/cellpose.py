import numpy as np
import xarray as xr

__all__ = [
    "setup_cellpose_zarrs",
    "make_cellpose_dataset",
]


def setup_cellpose_zarrs(group, img_shape):
    """
    Create a group in root and set up arrays to hold the outputs of cellpose
    (masks, probabilities, and flows).

    Parameters
    ----------
    group : zarr.group
        Group in the zarr store where cellpose outputs will be saved.
    img_shape : tuple[int, int, int]
        Shape corresponding to TYX dimensions of a single scene timelapse.

    Returns
    -------
    group : zarr.group
        Now contains cellpose output arrays
    """
    raise DeprecationWarning(
        "Recommended to use xr.Datasets for everything. See" "mu.cellpose.make_cellpose_dataset"
    )

    _ = group.zeros('masks', shape=img_shape, dtype='uint16', chunks=(1, -1, -1), overwrite=True)
    _ = group.zeros(
        'flows', shape=(img_shape[0], 2, *img_shape[1:]), chunks=(1, -1, -1), overwrite=True
    )
    _ = group.zeros('probs', shape=img_shape, chunks=(1, -1, -1), overwrite=True)
    _ = group.zeros('styles', shape=(img_shape[0], 256), chunks=(1, -1), overwrite=True)

    return group


def make_cellpose_dataset(masks, flows, styles):
    """
    Create a single position xr.Dataset for cellpose outputs.

    Parameters
    ----------
    masks : list[np.ndarray]
        First output of model predictions.
    flows : list[np.ndarray]
        Second output of model prediction. will be unpacked and
        stored in separate variables in the resulting dataset:
        cp_flows_y, cp_flows_x, and cp_probs.
    styles : list[np.ndarray]
        Style vector for each image.

    Returns
    -------
    cp_ds : xr.Dataset
        Dataset containing cellpose outputs.
    """
    cp_ds = xr.Dataset()
    cp_ds['cp_masks'] = xr.DataArray(np.stack(masks), dims=list('TYX'))
    cp_ds['cp_flows_y'] = xr.DataArray(np.stack([f[1][0] for f in flows]), dims=list('TYX'))
    cp_ds['cp_flows_x'] = xr.DataArray(np.stack([f[1][1] for f in flows]), dims=list('TYX'))
    cp_ds['cp_probs'] = xr.DataArray(np.stack([f[2] for f in flows]), dims=list('TYX'))
    cp_ds['cp_styles'] = xr.DataArray(np.stack(styles), dims=['T', 'style'])
    return cp_ds
