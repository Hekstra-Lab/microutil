import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import tifffile as tiff
import glob
import re
import xml.etree.ElementTree as ET

__all__ = [
            "get_standard_metadata",
            "get_ldm_metadata",
            "get_coords",
            "load_leica_frames",
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

def ldm_meta_split(x):
    name = x.acq_name
    fov, mode, ldm_idx = re.split(r"(\d+)", name)[1:4]
    fov = int(fov)-1
    mode = mode.strip("_")
    ldm_idx = int(ldm_idx)
    return pd.Series([fov, mode, ldm_idx],index=['fov', 'mode','ldm_idx'])


def get_ldm_metadata(data_dir, meta_tag="_Properties.xml"):
    """
    This is somewhat bespoke. But it will work as long as LDM jobs are titled according to F{fov#}_{mode}
    """
    metadata = pd.DataFrame(
        sorted(glob.glob(data_dir +"*"+ meta_tag)), columns=["filename"]
    )
    metadata["acq_name"] = metadata.filename.str.split("/").apply(
        lambda x: x[-1].replace(meta_tag, "")
    )
    metadata = metadata.join(metadata.apply(ldm_meta_split, axis=1, result_type='expand'))
    metadata = metadata.apply(gogogo_dimension_data, axis=1, result_type='expand')
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
            try:
                h, m, s = [float(x) for x in re.split(r"(\d+)h(\d+)m([0-9.]+)", a["Length"])[1:-1]]
                entry[f'{a["DimID"].lower()}_length'] = s + 60 * (m + 60 * h)
            except:
                entry[f'{a["DimID"].lower()}_length'] = None

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


def get_coords(meta_df, dims='STCZYX', others=None):
    """
    Generate xarray coordinates (STCZYX) from leica metadata stored
    in a pandas DataFrame by get_standard_metadata.

    Parameters
    ----------

    """
    coords = {}
    if 'S' in dims:
        coords['S'] = np.arange(meta_df.fov.nunique())
    for x in dims:
        if x!='S':
            length = meta_df[x.lower()+"_length"].iloc[0]
            elements = meta_df[x.lower()+"_elements"].iloc[0]
            if ~np.isnan(length) and ~np.isnan(elements):
                coords[x] = np.linspace(0,length,elements)
    if others is not None:
        coords = {**coords, **others}
    return coords


def leica_stczyx(x):
    l = re.split(r"(\d+)", x.filename.split("/")[-1])[1:-1:2]
    l.pop(1)
    # s = pd.Series(l, index=ordered).astype(int)
    s = pd.Series(l, index=list("STZC")).astype(int)
    s[0] -= 1
    return s

def leica_ldm_stczyx(x):
    l = re.split(r"(\d+)", x.filename.split("/")[-1])[1:-1:2]
    s = pd.Series(l, index=list("STZC")).astype(int)
    s[0] -= 1
    return s

def leica_ldm_stcrzyx(x):
    l = re.split(r"(\d+)",x.filename.split('/')[-1])[1:-1:2]
    s = pd.Series(l, index=list("STRZC")).astype(int)
    s[0]-= 1
    return s

def ldm_to_time(inds):
    mapper = pd.Series(dtype='uint16')
    for i,s in inds.groupby('S'):
        mapper = mapper.append(pd.Series(data = np.arange(s['T'].nunique()),
                      index=s['T'].unique()))
    #return mapper
    inds['T'] = mapper[inds['T'].values].values
    
def load_leica_frames(df, idx_mapper, coords=None, chunkby_dims='CZ'):

    if callable(idx_mapper):
        df = df.join(
            df.apply(idx_mapper, axis=1, result_type='expand') 
        )
    elif isinstance(idx_mapper, pd.DataFrame):
        df = df.join(idx_mapper)
    else:
        raise TypeError(
            "Must provide a callable to map names to indices or a pandas dataframe containing the indices"
        )

    #     ordered_cols = [df.columns[0]]+list('STCZ')
    #     df = df[ordered_cols]
    group_dims = [x for x in df.columns[1:] if x not in chunkby_dims]
    
    # if you end early there might not be the same number of frames in each pos
    # cutoff at the worst case scenario so things can be rectangular
    cutoffs = df.groupby('S').nunique().min().drop('filename')
    df = df.loc[
        (df.loc[:,~df.columns.isin(['S','filename'])] < cutoffs).all('columns')
    ]
    chunks = np.zeros(df[group_dims].nunique().values, dtype='object')
    
    for idx, val in df.groupby(group_dims):
        darr = da.from_zarr(tiff.imread(val.filename.tolist(), aszarr=True)).rechunk(-1)
        #shape = tuple(cutoffs[x] for x in  chunkby_dims) + darr.shape[-2:] 
        shape = tuple(x for i,x in cutoffs.iteritems() if i in chunkby_dims) + darr.shape[-2:]  
        #print(idx, shape)
        darr = darr.reshape(shape)
        chunks[idx] = darr
    
    chunks = np.expand_dims(chunks, tuple(range(-1, -len(chunkby_dims) - 3, -1)))
       
    d_data = da.block(chunks.tolist())
    x_data = xr.DataArray(
        d_data,
        dims=group_dims + [x for x in df.columns if x in chunkby_dims] + ['Y', 'X'],
    )
    if coords is not None:
        x_data = x_data.assign_coords(coords)
    x_data = x_data.transpose('S', 'T', 'C',..., 'Z', 'Y', 'X')
    return x_data

def load_srs_timelapse_dataset(data_dir):
    """
    This essentially assumes that files are named according to `Pos{S}_{mode}{ldm_idx}_t{R}_z{Z}_ch{C}.tif`
    """
    #glob the files
    srs_files = pd.DataFrame({"filename":sorted(glob.glob(data_dir+"*srs*z*.tif"))})
    fluo_files = pd.DataFrame({"filename":sorted(glob.glob(data_dir+"*fluo*z*.tif"))})
    
    #parse filenames -> indices
    srs_inds = srs_files.apply(leica_ldm_stcrzyx, axis=1, result_type='expand')
    ldm_to_time(srs_inds)
    fluo_inds = fluo_files.apply(leica_ldm_stczyx, axis=1, result_type='expand')
    ldm_to_time(fluo_inds)
   
    #parse metadata -> coords
    metadata = get_ldm_metadata(data_dir+"/Pos*")
    f_coords = get_coords(metadata.loc[metadata['mode']=='fluo'],'SZYX', {'C':['GFP', 'mCherry', 'BF']})
    s_coords = get_coords(metadata.loc[metadata['mode']=='srs'], 'SZYX',{'C':['BF', 'SRS']})
    
    #load the images
    srs_data = load_leica_frames(srs_files, srs_inds, coords=s_coords)
    fluo_data = load_leica_frames(fluo_files, fluo_inds, coords=f_coords)
    
    #combine into dataset and return
    return xr.Dataset({'srs':srs_data, 'fluo':fluo_data}).astype(srs_data.dtype)
