import tifffile
import dask.array as da
import xarray as xr
import numpy as np

__all__ = [
    "load_image_sequence",
    "load_mm_frames",
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


def load_mm_frames(data_dir, glob_pattern=None, chunkby_dims=['C', 'Z']):
    """
    Lazily read micromanager generated single frame tiff files.

    Parameters
    ----------
    data_dir : str or None
        Path to directory containing Pos{i} subdirectories
    glob_pattern : str or None
        Glob pattern to match files in a single directory. If None,
        all .tif files will be read
    chunkby_dims : list of str default ['C','Z']
        Dimensions to chunk resulting dask array by. X and Y dimensions
        will always be in a single chunk. Can contain any of S, T, C, or Z.

    Returns
    -------
    arr : xr.DataArray
        Unevaluated dask array containing all the files from the directory.
        Shape will vary but dimensions will always be (S, T, C, Z, Y, X)
    """
    if data_dir is None and glob_pattern is None:
        raise ValueError("Must specify data_dir or glob_pattern")

    position_dirs = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    if len(position_dirs) == 0 and glob_pattern is None:
        raise ValueError("No subdirectories found and no glob pattern provided")

    if len(position_dirs) > 0:
        for i, pos in enumerate(position_dirs):
            fseries = pd.Series(sorted(glob.glob(pos + '/*.tif')))
            df = pd.DataFrame({'filename': fseries})
            df[['C', 'S', 'T', 'Z']] = df.apply(
                lambda x: re.split(
                    'img_channel(\d+)_position(\d+)_time(\d+)_z(\d+).tif', x.filename
                )[1:-1],
                axis=1,
                result_type='expand',
            )

            if i == 0:
                all_files = df
            else:
                all_files = all_files.append(df)

            all_files[['C', 'T', 'S', 'Z']] = all_files[['C', 'T', 'S', 'Z']].astype(int)

    else:
        fseries = pd.Series(sorted(glob.glob(data_dir + glob_pattern)))
        df = pd.DataFrame({'filename': fseries})
        df[['C', 'S', 'T', 'Z']] = df.apply(
            lambda x: re.split('img_channel(\d+)_position(\d+)_time(\d+)_z(\d+).tif', x.filename)[
                1:-1
            ],
            axis=1,
            result_type='expand',
        )

        all_files = df
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
            darr = da.from_zarr(tiff.imread(val.filename.tolist(), aszarr=True)).rechunk(-1)
            shape = tuple(cutoffs[x] for x in chunkby_dims) + darr.shape[-2:]
            darr = darr.reshape(shape)
            chunks[idx] = darr

        chunks = np.expand_dims(chunks, tuple(range(-1, -len(chunkby_dims) - 3, -1)))

        d_data = da.block(chunks.tolist())
        x_data = xr.DataArray(
            da.block(chunks.tolist()), dims=group_dims + chunkby_dims + ['Y', 'X']
        )

        return x_data.transpose('S', 'T', 'C', 'Z', 'Y', 'X')
