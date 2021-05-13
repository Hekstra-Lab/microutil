__all__ = [
    "average",
]

import numpy as np
import xarray as xr
import warnings
import scipy.ndimage as ndi


def average(ds, intensity_name, label_name='labels', cell_dim_name="CellID", dims='STCZYX'):
    """
    Compute the average of the inntensity array over each labelled area.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    Nmax = labels.isel({T: -1}).max().item()
    intensity = ds[intensity_name]
    labels = ds[label_name]

    def padded_mean(intensity, labels):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(ndi.mean(intensity, labels=labels, index=np.arange(1, Nmax)))
        return out

    return xr.apply_ufunc(
        padded_mean,
        intensity,
        labels,
        input_core_dims=[[Y, X], [Y, X]],
        output_core_dims=[[cell_dim_name]],
        vectorize=True,
    )
