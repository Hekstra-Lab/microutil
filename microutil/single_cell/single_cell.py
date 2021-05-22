__all__ = [
    "average",
    "center_of_mass",
    "cell_op",
    "bootstrap",
]

import numpy as np
import pandas as pd
import xarray as xr
import warnings
import scipy.ndimage as ndi


def cell_op(
    ds,
    func,
    intensity,
    exclude_dims=None,
    output_core_dims=None,
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
    exclude_dims : Iterable of str or None, default None
        Names of dimensions that are allowed to change size during apply_ufunc.
    output_core_dims : list of str or None, default None
        Output core dims to pass to apply_ufunc. cell_dim_name is handled automatically.
    label_name : str, default 'labels'
        Name for the dimension containing the individual cell labels in ds.
    cell_dim_name : str, default "CellID"
        Name for the dimension containing the data for each individual cell in the output array.
    dims: str or list of str, default 'STCZYX`
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

    else:
        output_core_dims = [[cell_dim_name] + output_core_dims]

    if isinstance(intensity, str):
        intensity = ds[intensity]

    if intensity is None:
        intensity = xr.ones_like(labels)

    Nmax = labels.isel({T: -1}).max().item()

    return xr.apply_ufunc(
        func,
        intensity,
        labels,
        kwargs={'Nmax': Nmax},
        input_core_dims=[[Y, X], [Y, X]],
        exclude_dims=exclude_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
    )


def average(ds, intensity, label_name='labels', cell_dim_name="CellID", dims='STCZYX'):
    """
    Compute the average of the inntensity array over each labelled area.

    Parameters
    ----------

    dims: str or list of str, default 'STCZYX`
          Dimensions names for `bf` that correspond to STCZYX
    """

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
    dims: str or list of str, default 'STCZYX`
          Dimensions names for `ds` that correspond to STCZYX

    Returns
    -------
    coms : xarray.DataArray
        Dataarray containing the center of mass of each labelled cell in ds.
        Same shape and dims as ds except for YX which are replaced by cell_dim_name.
    """
    # TODO rescale com values according to XY coordinates of ds
    # TODO low priority - write helper function somwhere for scattering coms on hyperslicer

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
    dims: str or list of str, default 'STCZYX`
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
