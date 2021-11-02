__all__ = [
    "remove_holes_and_objs",
    "df_to_normalized_arr",
    "get_label_area_arrs",
]

import numpy as np
import xarray as xr
import dask.array as da
from skimage.morphology import remove_small_holes, remove_small_objects


def remove_holes_and_objs(im, size):
    return remove_small_objects(remove_small_holes(im, size), size)


def df_to_normalized_arr(df, train_columns):
    train_df = df[train_columns]
    X = train_df - train_df.mean(0)
    std_devs = train_df.std(0)
    std_devs[std_devs == 0] = 1
    X = X / std_devs
    X = X[train_columns].values
    return X


def get_label_area_arrs(
    df, ds, as_dask=True, dims=list('STCZYX'), label_name='label', area_name='area'
):
    S, T, C, Z, Y, X = dims
    frame_gb = df.groupby([S, T])
    max_cells = df.nunique()[label_name].max()
    frame_shape = df.nunique()[[S, T]].values
    check_labels = np.zeros((*frame_shape, max_cells), dtype='uint16')
    check_areas = np.zeros_like(check_labels)
    for idx, sub_df in frame_gb:
        n_cells = sub_df.nunique()[label_name]
        check_labels[(*idx, slice(0, n_cells))] = sub_df[label_name]
        check_areas[(*idx, slice(0, n_cells))] = sub_df[area_name]
    if as_dask:
        check_labels = da.from_array(check_labels)
        check_areas = da.from_array(check_areas)
    check_labels = xr.DataArray(check_labels, dims=[S, T, 'cells'], coords={T: ds[T]})
    check_areas = xr.DataArray(check_areas, dims=[S, T, 'cells'], coords={T: ds[T]})
    return check_labels, check_areas
