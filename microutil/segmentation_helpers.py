__all__ = [
    "remove_holes_and_objs",
    "df_to_normalized_arr",
    "get_label_arr",
    "norm_label2rgb",
    "make_show_labels",
]

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from skimage.color import label2rgb
from skimage.morphology import remove_small_holes, remove_small_objects


def remove_holes_and_objs(im, size):
    """
    Remove small object and holes from a binary mask in one function.

    Parameters
    ----------
    im : np.array of Bool or int
        Mask array
    size: int
        Size in pixels below which an object/hole is considered small.

    Returns
    -------
    cleaned_mask : np.array with same shape as im
    """

    return remove_small_objects(remove_small_holes(im, size), size)


def df_to_normalized_arr(df, train_columns=None):
    """
    Normalize the columns of a dataframe by subtracting the mean and dividing by the standard deviation.
    Helpful if you want to pass a dataframe into sklearn type functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing un-normalized data
    train_colums : list[str] or None default None
        Columns of df to keep for the normalization process. If none, keep all the columns.
    """
    if train_columns is not None:
        train_df = df[train_columns]

    else:
        train_df = df

    X = train_df - train_df.mean(0)
    std_devs = train_df.std(0)
    std_devs[std_devs == 0] = 1
    X = X / std_devs
    X = X[train_columns].values
    return X


def get_label_arr(df, ds, as_dask=True, dims=list('STCZYX'), label_name='label'):
    """
    Return the cell labels column of df in an array that can be aligned with ds.
    If df has a MultiIndex, we will look for S, T, and label_name there. Otherwise
    we expect they are columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing cell labels and S,T indices.
    ds : xr.Dataset
        Dataset with which cell labels need to align
    as_dask : bool default True
        Whether to return the data as a dask backed DataArray
    dims : list[str] default list("STCZYX")
        Dimension names for the six standard dimensions of microscopy datasets.
    label_name : str default "label"
        Name of the column in df that contains cell label id numbers.

    Returns
    -------
    return check_labels : xr.DataArray
        DataArray aligned with non-spatial dimensions of ds that contains the
        labels of the cells in each FOV padded with zeros for positions that have
        less than the maximum number of cells.
    """

    S, T, C, Z, Y, X = dims
    if S in df and T in df and label_name in df:
        frame_gb = df.groupby([S, T])
        max_cells = df.nunique()[label_name].max()
        frame_shape = (ds.sizes[S], ds.sizes[T])
        check_labels = np.zeros((*frame_shape, max_cells), dtype='uint16')
        # check_areas = np.zeros_like(check_labels)
        for idx, sub_df in frame_gb:
            n_cells = sub_df.nunique()[label_name]
            check_labels[(*idx, slice(0, n_cells))] = sub_df[label_name]
    elif isinstance(df.index, pd.MultiIndex):
        frame_gb = df.groupby(level=[S, T])
        max_cells = df.index.get_level_values(label_name).max()
        frame_shape = (ds.sizes[S], ds.sizes[T])
        check_labels = np.zeros((*frame_shape, max_cells), dtype='uint16')
        # check_areas = np.zeros_like(check_labels)
        for idx, sub_df in frame_gb:
            n_cells = sub_df.index.get_level_values(label_name).nunique()
            check_labels[(*idx, slice(0, n_cells))] = sub_df.index.get_level_values(
                label_name
            ).values
    else:
        raise ValueError(f"Unable to find Scene ({S}) or Time ({T}) in df")

    if as_dask:
        check_labels = da.from_array(check_labels)
    check_labels = xr.DataArray(check_labels, dims=[S, T, 'cells'], coords={T: ds[T]})
    return check_labels


def norm_label2rgb(labels):
    """
    Lightly wrap skimage.colors.label2rgb to return floats in [0,1)
    rather than ints for display in matplotlib without warnings.

    Parameters
    ----------
    labels : np.array of int
        Array containing labelled regions with background having value 0.

    Returns
    -------
    show_labels: np.array of float with shape=labels.shape+(3,)
    """
    show = label2rgb(labels, bg_label=0)
    c_max = show.max((0, 1))
    return show / c_max


def make_show_labels(
    ds, label_name='labels', show_label_name='show_labels', rgb_dim_name='rgb', dims=list('STCZYX')
):
    """
    Add a new variable to ds that contains label regions colored with norm_label2rb.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing labelled images
    label_name: str default "labels"
        Name of labelled image variable in ds.
    show_label_name: str default "show_labels"
        Name of the new variable that will contain the colored label images.
    rgb_dim_name: str default 'rgb'
        Name of the dim that will contain the color channels.
    dims: str or list of str default list('STCZYX')
        Names of standard dims in the dataset. Color will be assigned
        to labels in each YX frame independently

    Returns
    -------
    Nothing. ds is updated inplace.
    """

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    ds[show_label_name] = xr.apply_ufunc(
        norm_label2rgb,
        ds[label_name],
        input_core_dims=[[Y, X]],
        output_core_dims=[[Y, X, rgb_dim_name]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={'output_sizes': {rgb_dim_name: 3}},
    )
