import tifffile
import dask.array as da
import xarray as xr
import numpy as np

__all__ = [
    "load_image_sequence",
]


def load_image_sequence(filenames, z_centered=True, pattern=None):
    """
    Load an image sequence from micromanager .ome.tif files.
    Loads as zarr into dask into xarray

    Parameters
    ----------
    filenames : str
        The path/regex that indentifies the file names.
    z_centered : bool, default: True
        Whether to offset
    pattern : str or None, default: None
        Regex to match the sequence. If None
        this will default ot a Regex that matches: Pos[position number]

    Returns
    -------
    xarray.DataArray
        The tiffs loaded as an Xarray with coords filled in.
    """
    if pattern is None:
        pattern = r"""(?ix)
        _?(?:(Pos)(\d{1,4}))
        """

    t = tifffile.TiffSequence(filenames, pattern=pattern)
    # load the first file to grab the metadata
    meta = tifffile.TiffFile(t.files[0]).micromanager_metadata
    # load axis order but extract the positions from the list
    # also reverse it because that's what we need ¯\_(ツ)_/¯
    # call `list` on it so that we aren't modifying the original metadata
    ax_order = list(meta["Summary"]["AxisOrder"])
    ax_order.remove("position")
    ax_order = ax_order[::-1]

    channel_names = meta["Summary"]["ChNames"]

    z_step = meta["Summary"]["z-step_um"]
    n_slices = meta["Summary"]["Slices"]
    Z = np.linspace(0, n_slices * z_step, n_slices)

    if z_centered:
        if len(Z) % 2 == 0:
            raise ValueError(
                f"There are an even number of z points ({len(Z)}) so z_centered cannot be True"
            )
        Z -= Z[int(len(Z) / 2 - 0.5)]

    # n_times = meta["Summary"]["Frames"]
    arr = da.from_zarr(t.aszarr())
    n_times = arr.shape[1]

    # Nominal timepoints in ms
    Times = np.linspace(0, meta["Summary"]["Interval_ms"] * n_times, n_times)

    arr = xr.DataArray(
        arr,
        dims=("pos", "time", "channel", "z", "y", "x"),
        coords={"channel": channel_names, "z": Z, "time": Times},
        attrs={"Summary": meta["Summary"], "Comment": meta["Comments"]["Summary"]},
    ).transpose(..., "x", "y")
    return arr
