import glob
import json
import os
import re

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr

__all__ = [
    "micromanager_metadata_to_coords",
    "load_image_sequence",
    "load_mm_frames",
]


def micromanager_metadata_to_coords(summary, n_times=None, z_centered=True):
    """
    Given the 'Summary' dict from micromanager, parse the information
    into coords for a corresponding DataArray.

    Parameters
    ----------
    summary : dict
        Micromanager metadata dictionary.
    n_times : int
        Number of actual time points in dataset. If None, read the 'Frames'
        attribute from metadata. This will cause problems if the experiment
        stopped short of the desired number of timepoints.
    z_centered : bool, default True
        Rescale the z coordinate to be relative to the center slice.
        Will only work with an odd number of slices.

    Returns
    -------
    coords : dict

    """
    # load axis order but extract the positions from the list
    # also reverse it because that's what we need ¯\_(ツ)_/¯
    # call `list` on it so that we aren't modifying the original metadata
    ax_order = list(summary["AxisOrder"])
    ax_order.remove("position")
    ax_order = ax_order[::-1]

    coords = {}

    channel_names = summary["ChNames"]
    coords['C'] = channel_names

    z_step = summary["z-step_um"]
    n_slices = summary["Slices"]
    z = np.linspace(0, n_slices * z_step, n_slices)

    if z_centered:
        if len(z) % 2 == 0:
            raise ValueError(
                f"There are an even number of z points ({len(z)}) so z_centered cannot be True"
            )
        z -= z[int(len(z) / 2 - 0.5)]

    coords['Z'] = z

    if n_times is None:
        n_times = summary["Frames"]

    # Nominal timepoints in ms
    times = np.linspace(0, summary["Interval_ms"] * n_times, n_times)
    coords['T'] = times

    return coords


def load_image_sequence(filenames, z_centered=True, pattern=None, coords=None):
    """
    Load an image sequence from micromanager .ome.tif files.
    Loads as zarr into dask into xarray

    Parameters
    ----------
    filenames : str
        The path/regex that identifies the file names.
    z_centered : bool, default: True
        Whether to offset
    pattern : str or None, default: None
        Regex to match the sequence. If None
        this will default to a Regex that matches: Pos[position number]
    coords : None or dict
        Dictionary containing coordinates for the final DataArray. Must have
        'S', 'T', 'C', 'Z', 'Y, and 'X' as keys. If None
        attempt to read the relevant information from micromanager metadata.

    Returns
    -------
    xarray.DataArray
        The tiffs loaded as an Xarray with coords filled in.
    """
    if pattern is None:
        pattern = r"""(?ix)
        _?(?:(Pos)(\d{1,4}))
        """

    # load the first file to grab the metadata
    t = tifffile.TiffSequence(filenames, pattern=pattern)
    arr = da.from_zarr(t.aszarr())
    n_times = arr.shape[1]

    if coords is None:
        meta = tifffile.TiffFile(t.files[0]).micromanager_metadata
        coords = micromanager_metadata_to_coords(
            meta['Summary'], n_times=n_times, z_centered=z_centered
        )

    arr = xr.DataArray(
        arr,
        dims=("S", "T", "C", "Z", "Y", "X"),
        coords=coords,
        attrs={"Summary": meta["Summary"], "Comment": meta["Comments"]["Summary"]},
    )
    return arr


def load_mm_frames(
    data_dir: str,
    position_folder_regex: str = "Pos[0-9]+",
    filename_regex=r".+\.tif{1,2}",
    chunkby_dims=['C', 'Z'],
    z_centered=True,
    coords=None,
):
    """
    Lazily read micromanager generated single frame tiff files.

    Parameters
    ----------
    data_dir : str or None
        Path to directory containing Pos{i} subdirectories
    position_folder_regex : str, default: "Pos[0-9]+"
        A regular expression to match the folder names specifying positions.
        E.g. to match Pos0, Pos1, and OtherName0, OtherName1 pass
        "(Pos|OtherName)[0-9]+"
    filename_regex : str or None
        Regular expression to identify which tiff files to read. If None then
        all .tif[f] files will be read.
    chunkby_dims : list of str default ['C','Z']
        Dimensions to chunk resulting dask array by. X and Y dimensions
        will always be in a single chunk. Can contain any of S, T, C, or Z.
    z_centered : bool default true
        Whether or not to use Z coordinates relative to the center slice.
    coords : None or dict
        Dictionary containing coordinates for the final DataArray. Must have
        'S', 'T', 'C', 'Z', 'Y, and 'X' as keys. If None
        attempt to read the relevant information from micromanager metadata.

    Returns
    -------
    arr : xr.DataArray
        Unevaluated dask array containing all the files from the directory.
        Shape will vary but dimensions will always be (S, T, C, Z, Y, X)
    """

    pos_reg = re.compile(f"{data_dir.rstrip('/')}/" + position_folder_regex)
    fname_reg = re.compile(filename_regex)

    position_dirs = sorted(
        f.path for f in os.scandir(data_dir) if f.is_dir() and pos_reg.match(f.path)
    )

    def dir_to_df(dir):
        fseries = pd.Series(sorted(filter(fname_reg.match, glob.glob(dir.rstrip("/") + "/*"))))
        df = pd.DataFrame({'filename': fseries})
        df[['C', 'S', 'T', 'Z']] = df.apply(
            lambda x: re.split(r'img_channel(\d+)_position(\d+)_time(\d+)_z(\d+).tif', x.filename)[
                1:-1
            ],
            axis=1,
            result_type='expand',
        )
        return df

    if len(position_dirs) > 0:
        for i, pos in enumerate(position_dirs):
            fseries = pd.Series(sorted(filter(fname_reg.match, glob.glob(pos + "/*"))))
            df = pd.DataFrame({'filename': fseries})
            df[['C', 'S', 'T', 'Z']] = df.apply(
                lambda x: re.split(
                    r'img_channel(\d+)_position(\d+)_time(\d+)_z(\d+).tif', x.filename
                )[1:-1],
                axis=1,
                result_type='expand',
            )
            if i == 0:
                all_files = df
            else:
                all_files = pd.concat([all_files, df])

    else:
        all_files = dir_to_df(data_dir)

    if len(df) == 0:
        raise ValueError("No files found")

    all_files[['C', 'T', 'S', 'Z']] = all_files[['C', 'T', 'S', 'Z']].astype(int)

    # if you end early there might not be the same number of frames in each pos
    # cutoff at the worst case scenario so things can be rectangular
    cutoffs = all_files.groupby('S').nunique().min().drop('filename')

    use_files = all_files.loc[
        all_files.apply(lambda x: (x[['C', 'T', 'Z']] < cutoffs).all(), axis=1)
    ]

    group_dims = [x for x in ['S', 'T', 'C', 'Z'] if x not in chunkby_dims]

    chunks = np.zeros(use_files[group_dims].nunique().values, dtype='object')

    for idx, val in use_files.groupby(group_dims):
        darr = da.from_zarr(tifffile.imread(val.filename.tolist(), aszarr=True)).rechunk(-1)
        shape = tuple(cutoffs[x] for x in chunkby_dims) + darr.shape[-2:]
        darr = darr.reshape(shape)
        chunks[idx] = darr

    chunks = np.expand_dims(chunks, tuple(range(-1, -len(chunkby_dims) - 3, -1)))

    da.block(chunks.tolist())

    x_data = xr.DataArray(da.block(chunks.tolist()), dims=group_dims + chunkby_dims + ['Y', 'X'])

    if coords is None:
        with open(position_dirs[0] + "/metadata.txt") as f:
            meta = json.load(f)

        coords = micromanager_metadata_to_coords(
            meta['Summary'], n_times=x_data['T'].values.shape[0], z_centered=z_centered
        )

    try:
        x_data = x_data.assign_coords(coords)
    except ValueError as e:
        # can happen if you ignore one of the channels - then the metadata
        # won't match up with actual shape of the array
        print(f'Unable to assign coords due to this error:\n{e}')

    return x_data.transpose('S', 'T', 'C', 'Z', 'Y', 'X')
