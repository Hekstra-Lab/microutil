import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import tifffile as tiff

__all__ = [
            "get_standard_metadata",
            "get_coords",
            "load_standard_leica_frames",
            "gogogo_dimension_data",
            "leica_stczyx",
]

META_COLS = [
    "timestamp",
    "x_origin",
    "y_origin",
    "z_origin",
    "t_origin",
    "x_length",
    "y_length",
    "z_length",
    "t_length",
    "x_elements",
    "y_elements",
    "z_elements",
    "t_elements",
    "stage_x",
    "stage_y",
    "stage_z",
    "channels",
]


def get_standard_metadata(data_dir, meta_tag="_Properties.xml"):
    """
    Get metadata from and for leica single frame tifs.

    Parameters
    ----------

    """
    metadata = pd.DataFrame(sorted(glob.glob(data_dir + '*' + meta_tag)), columns=["filename"])
    metadata["acq_name"] = metadata.filename.apply(
        lambda x: re.split(r"_Properties.xml", x.split('/')[-1])[0]
    )

    metadata[META_COLS] = None
    metadata = metadata.apply(gogogo_dimension_data, axis=1)
    metadata['fov'] = metadata.apply(
        lambda x: int(re.split(r"Pos(\d+)", x.filename)[1]) - 1, axis=1
    )
    return metadata


def gogogo_dimension_data(entry):
    parsed = ET.parse(entry.filename)
    for x in parsed.iter("TimeStampList"):
        d, t, ms = x.attrib.values()
    entry.timestamp = pd.to_datetime(d + " " + t) + pd.to_timedelta(int(ms), unit="ms")
    for d in parsed.iter("DimensionDescription"):
        a = d.attrib
        entry[f'{a["DimID"].lower()}_origin'] = float(re.sub("[^0-9.]", "", a["Origin"]))
        if a["DimID"] == "T":
            h, m, s = [float(x) for x in re.split(r"(\d+)h(\d+)m([0-9.]+)", a["Length"])[1:-1]]
            entry[f'{a["DimID"].lower()}_length'] = s + 60 * (m + 60 * h)

        else:
            entry[f'{a["DimID"].lower()}_length'] = float(re.sub("[^0-9.]", "", a["Length"]))
        entry[f'{a["DimID"].lower()}_elements'] = int(re.sub("[^0-9.]", "", a["NumberOfElements"]))
    for d in parsed.iter("FilterSettingRecord"):
        a = d.attrib
        if a["ClassName"] == "CXYZStage" and "DM6000 Stage Pos" in a["Description"]:
            entry[f"stage_{a['Attribute'].strip('Pos').lower()}"] = float(a["Variant"])
    for f in parsed.iter("FrameCount"):
        entry["channels"] = int(f.text.split()[1].strip("()"))
    return entry


def get_coords(meta_df, C_names):
    """
    Generate xarray coordinates (STCZYX) from leica metadata stored
    in a pandas DataFrame by get_standard_metadata.

    Parameters
    ----------

    """
    coords = {}
    coords['S'] = np.arange(meta_df.fov.nunique())
    coords['T'] = np.linspace(0, meta_df.t_length.iloc[0], meta_df.t_elements.iloc[0])
    coords['Z'] = np.linspace(0, meta_df.z_length.iloc[0], meta_df.z_elements.iloc[0])
    coords['C'] = C_names
    coords['X'] = np.linspace(0, meta_df.x_length.iloc[0], meta_df.x_elements.iloc[0])
    coords['Y'] = np.linspace(0, meta_df.y_length.iloc[0], meta_df.y_elements.iloc[0])
    return coords


def leica_stczyx(x):
    l = re.split(r"(\d+)", x.filename.split("/")[-1])[1:-1:2]
    l.pop(1)
    # s = pd.Series(l, index=ordered).astype(int)
    s = pd.Series(l, index=list("STZC")).astype(int)
    s[0] -= 1
    return s


def load_standard_leica_frames(df, idx_mapper, coords=None, chunkby_dims='CZ'):

    if callable(idx_mapper):
        df = df.merge(
            df.apply(idx_mapper, axis=1, result_type='expand'), right_index=True, left_index=True
        )
    elif isinstance(idx_mapper, pd.DataFrame):
        df = df.merge(idx_mapper)
    else:
        raise TypeError(
            "Must provide a callable to map names to indices or a pandas dataframe containing the indices"
        )

    #     ordered_cols = [df.columns[0]]+list('STCZ')
    #     df = df[ordered_cols]
    group_dims = [x for x in df.columns[1:] if x not in chunkby_dims]
    chunks = np.zeros(df[group_dims].nunique().values, dtype='object')

    for idx, val in df.groupby(group_dims):
        darr = da.from_zarr(tiff.imread(val.filename.tolist(), aszarr=True)).rechunk(-1)
        shape = tuple(df[x].nunique() for x in df.columns if x in chunkby_dims) + darr.shape[-2:]
        print(idx)
        print(darr.shape)
        print(shape)
        darr = darr.reshape(shape)
        chunks[idx] = darr

    chunks = np.expand_dims(chunks, tuple(range(-1, -len(chunkby_dims) - 3, -1)))

    d_data = da.block(chunks.tolist())

    x_data = xr.DataArray(
        da.block(chunks.tolist()),
        dims=group_dims + [x for x in df.columns if x in chunkby_dims] + ['Y', 'X'],
    )
    print(group_dims + list(chunkby_dims) + ['Y', 'X'])
    print(x_data.shape)
    if coords is not None:
        x_data.assign_coords(coords)
    x_data = x_data.transpose('S', 'T', 'C', 'Z', 'Y', 'X')
    return x_data
