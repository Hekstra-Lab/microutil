__all__ = [
    "average",
    "center_of_mass",
    "cell_op",
]

import numpy as np
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
    print(f"{intensity.sizes=}")
    print(f"{labels.sizes=}")

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

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    def padded_com(intensity, labels, Nmax=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(ndi.center_of_mass(intensity, labels=labels, index=np.arange(1, Nmax)))
        return out

    return cell_op(
        ds,
        padded_com,
        None,
        output_core_dims=[com_name],
        label_name=label_name,
        cell_dim_name=cell_dim_name,
        dims=dims,
    )


# def bootstrap
