__all__ = [
    "average",
    "center_of_mass",
    "area",
    "cell_op",
    "bootstrap",
    "regionprops",
    "regionprops_df",
]

import warnings
from itertools import product
import os

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr
from dask import delayed
import dask.dataframe as dd
from skimage.measure import regionprops_table
import numba


def cell_op(
    ds,
    func,
    intensity,
    Nmax=None,
    exclude_dims=None,
    output_core_dims=None,
    output_sizes=None,
    label_name='labels',
    cell_dim_name='CellID',
    dims='STCZYX',
):
    """
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the labels
    func : callable
        Function provided to apply_ufunc. Should consume 2D numpy arrays.
    intensity : xr.DataArray, str, or None,
        Data array or variable name from which to draw the samples. Should align with ds[label_name].
        If None, an array of ones like ds[label_name] will be used.
    Nmax : int or None, default None
        Maximum number of cells in a single frame. Used to pad single frame results to coerce
        results into array-like shape. If None, the labels Variable in ds will be loaded into
        memory and the maximum
    exclude_dims : Iterable of str or None, default None
        Names of dimensions that are allowed to change size during apply_ufunc.
    output_core_dims : list of str or None, default None
        Output core dims to pass to apply_ufunc. cell_dim_name is handled automatically.
    output_sizes : iterable of int, default None
        Sizes of output core dims, only necessary
    label_name : str, default 'labels'
        Name for the dimension containing the individual cell labels in ds.
    cell_dim_name : str, default "CellID"
        Name for the dimension containing the data for each individual cell in the output array.
    dims : str or list of str, default 'STCZYX`
        Dimensions names for `ds` that correspond to STCZYX

    Returns
    -------
    out : xr.DataArray
        Dataarray containing the results of func applied to each labelled cell in ds.
        Shape and dims will depend on func.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    labels = ds[label_name]

    if Nmax is None:
        Nmax = labels.max().load().item() + 1
    else:
        Nmax += 1  # silly but makes the final array have CellID dim with size Nmax

    if exclude_dims is not None:
        tmp_set = set()
        for x in exclude_dims:
            if x in dims:
                tmp_set.add(x)
            else:
                warnings.warn("Supplied dim to exclude is not in dims")
        exclude_dims = tmp_set

    else:
        exclude_dims = set()

    if output_core_dims is None:
        output_core_dims = [[cell_dim_name]]
        output_sizes = {cell_dim_name: Nmax - 1}  # we dont return values for the background

    else:
        output_core_dims = [[cell_dim_name] + output_core_dims]
        if output_sizes is not None:
            output_sizes = dict(zip(output_core_dims, (Nmax, *output_sizes)))
        else:
            output_sizes = {cell_dim_name: Nmax}

    dask_gufunc_kwargs = {"output_sizes": output_sizes, "allow_rechunk": True}

    if isinstance(intensity, str):
        intensity = ds[intensity]

    if intensity is None:
        intensity = xr.ones_like(labels)

    return xr.apply_ufunc(
        func,
        intensity,
        labels,
        kwargs={'Nmax': Nmax},
        input_core_dims=[[Y, X], [Y, X]],
        exclude_dims=exclude_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
    )


def area(ds, Nmax=None, label_name='labels', cell_dim_name='CellID', dims='STCZYX'):
    """
    Compute the area of each labelled region in each frame.

    """

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    def padded_area(intensity, labels, Nmax=None):
        _, areas = np.unique(labels, return_counts=True)
        areas = areas[1:]
        out = np.pad(
            areas.astype(float), (0, Nmax - len(areas) - 1), "constant", constant_values=np.nan
        )
        return out

    areas = cell_op(
        ds,
        padded_area,
        None,
        Nmax=Nmax,
        label_name=label_name,
        cell_dim_name=cell_dim_name,
        dims=dims,
    )

    return areas


def average(ds, intensity, label_name='labels', cell_dim_name="CellID", dims='STCZYX'):
    """
    Compute the average of the inntensity array over each labelled area.

    Parameters
    ----------
    dims : str or list of str, default 'STCZYX`
        Dimensions names for `bf` that correspond to STCZYX
    """

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    def padded_mean(intensity, labels, Nmax=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(ndi.mean(intensity, labels=labels, index=np.arange(1, Nmax)))
        return out

    return cell_op(
        ds, padded_mean, intensity, label_name=label_name, cell_dim_name=cell_dim_name, dims=dims
    )


def center_of_mass(ds, com_name='com', label_name='labels', cell_dim_name='CellID', dims='STCZYX'):
    """
    Compute the center of mass of each labeled cell in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the labels
    com_name : str, default 'com'
        Name for the dimension containing the coordinates of the center of mass in the output array.
    label_name : str, default 'labels'
        Name for the dimension containing the individual cell labels in ds.
    cell_dim_name : str, default "CellID"
        Name for the dimension containing the data for each individual cell in the output array.
    dims : str or list of str, default 'STCZYX`
        Dimensions names for `ds` that correspond to STCZYX

    Returns
    -------
    coms : xarray.DataArray
        Dataarray containing the center of mass of each labelled cell in ds.
        Same shape and dims as ds except for YX which are replaced by cell_dim_name.
    """
    # TODO This should really take an intensity field too - like with cytosolic
    #      fluorescence we could estimate the actual center of mass rather than the
    #      centroid of the mask
    # TODO rescale com values according to XY coordinates of ds
    # TODO low priority - write helper function for scattering coms on hyperslicer

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    def padded_com(intensity, labels, Nmax=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(ndi.center_of_mass(intensity, labels=labels, index=np.arange(1, Nmax)))
        return out

    coms = cell_op(
        ds,
        padded_com,
        None,
        output_core_dims=[com_name],
        label_name=label_name,
        cell_dim_name=cell_dim_name,
        dims=dims,
    )

    coms[com_name] = [Y, X]

    return coms


def bootstrap(
    ds,
    intensity,
    n_samples,
    label_name='labels',
    cell_dim_name='CellID',
    sample_name='samples',
    dims='STCZYX',
):
    """
    Return bootstrap samples from each labelled cell in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the labels
    intensity : xr.DataArray or str
        Data array or variable name from which to draw the samples. Should align with ds[label_name].
    n_samples : int
        Number of samples to draw from each individual cell.
    label_name : str, default 'labels'
        Name for the dimension containing the individual cell labels in ds.
    cell_dim_name : str, default "CellID"
        Name for the dimension containing the data for each individual cell in the output array.
    sample_name : str, default 'samples'
        Name for dimension containing the bootstrap samples in the output array.
    dims : str or list of str, default 'STCZYX`
        Dimensions names for `ds` that correspond to STCZYX

    Returns
    -------
    bootstrapped : xr.DataArray
        Dataarray containing the samples from each labelled cell in ds.
        Same shape and dims as ds except for YX which are replaced by cell_dim_name.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    if isinstance(intensity, str):
        intensity = ds[intensity]

    Nmax = ds[label_name].max().item()

    rng = np.random.default_rng()

    bootstrapped = (
        xr.full_like(intensity.isel(X=0, Y=0).drop([Y, X]), np.nan, dtype=float)
        .expand_dims({cell_dim_name: Nmax + 1, sample_name: n_samples})
        .copy(deep=True)
    )

    for i in range(Nmax + 1):
        indexer = pd.DataFrame(
            np.array(np.nonzero(ds[label_name].data == i)).T, columns=list('STYX')
        )
        out_idx = {cell_dim_name: i}
        for group, vals in indexer.groupby([S, T]):
            # print(group, vals.shape[0])
            out_idx = {**out_idx, **dict(zip([S, T], group))}
            idx = rng.integers(vals.shape[0], size=n_samples)
            sample_idx = vals.iloc[idx].reset_index(drop=True).to_xarray()
            bootstrapped[out_idx] = intensity[sample_idx]

            # sample_ds = ds[sample_idx] # group+(slice(None),i)
            # for var in sample_ds:
            #    bootstrapped[var][out_idx] = sample_ds[var]
    return bootstrapped


DEFAULT_PROPERTIES = (
    'label',
    'bbox',
    'area',
    'convex_area',
    'eccentricity',
    'equivalent_diameter',
    'euler_number',
    'extent',
    'feret_diameter_max',
    'perimeter_crofton',
    'solidity',
    'moments_hu',
)


def regionprops_df(im, props, other_cols):
    df = pd.DataFrame(regionprops_table(im, properties=props))
    for k, v in other_cols.items():
        df[k] = v
    return df


def regionprops(ds, properties=DEFAULT_PROPERTIES, label_name='labels', dims='STCZYX'):
    """
    Loop over the frames of ds and compute the regionprops for
    each labelled image in each frame.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    d_regionprops = delayed(regionprops_df)

    loop_dims = {k: v for k, v in ds.labels.sizes.items() if k not in [Y, X]}

    all_props = []

    for dims in product(*(range(v) for v in loop_dims.values())):
        other_cols = dict(zip(loop_dims.keys(), dims))
        frame_props = d_regionprops(ds[label_name].data[dims], properties, other_cols)
        all_props.append(frame_props)

    cell_props = dd.from_delayed(all_props, meta=all_props[0].compute())
    cell_props = cell_props.repartition(os.cpu_count() // 2)  # .compute()

    return cell_props


def regionprops_pandas(ds, properties=DEFAULT_PROPERTIES, label_name='labels', dims='STCZYX'):
    """
    Loop over the frames of ds and compute the regionprops for
    each labelled image in each frame.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    def regionprops_df(im, props, other_cols):
        df = pd.DataFrame(regionprops_table(im, properties=props))
        for k, v in other_cols.items():
            df[k] = v
        return df

    loop_dims = {k: v for k, v in ds.labels.sizes.items() if k not in [Y, X]}

    all_props = []

    for dims in product(*(range(v) for v in loop_dims.values())):
        other_cols = dict(zip(loop_dims.keys(), dims))
        frame_props = regionprops_df(ds[label_name].data[dims], properties, other_cols)
        all_props.append(frame_props)

    # cell_props = dd.from_delayed(all_props, meta=all_props[0].compute())
    # cell_props.repartition(os.cpu_count()//2)#.compute()
    cell_props = pd.concat(all_props)
    return cell_props
