from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr
from aicsimageio.readers import TiffGlobReader
from aicsimageio.types import PathLike

__all__ = [
    "find_last_complete_timepoint",
    "index_complete_timepoints",
    "indexer_maker",
    "load_tiffs",
]


def find_last_complete_timepoint(df: pd.Dataframe) -> int:
    """
    Search an index dataframe for the last complete timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        An aicsimageio index dataframe with at least columns S, T, Z

    Returns
    -------
    last_complete_time_point : int
    """
    max_T = df["T"].max()
    max_S = df["S"].max()
    max_Z = df["Z"].max()
    for t in range(max_T, 0, -1):
        timepoint = df.loc[df["T"] == t].max()
        if not (timepoint.S == max_S) and (timepoint.Z == max_Z):
            continue
        return t


def index_complete_timepoints(df: pd.Dataframe) -> pd.Series:
    """
    Get an index filtering out incomplete timepoints

    Parameters
    ----------
    df : pd.DataFrame
        An aicsimageio index dataframe with at least columns S, T, Z

    Returns
    -------
    index : pd.Series of bool
    """
    last_time = find_last_complete_timepoint(df)
    return df["T"] <= last_time


def indexer_maker(index_names: str | Iterable[str] = 'TSCZ') -> callable:
    """
    Generate an indexer function for aicsimageio

    Parameters
    ----------
    index_names : str, or Iterable[str]
        The index names to use. Defaults to the order used by raman-mda-engine

    Returns
    -------
    indexer : function
        Function to pass to TiffGlobReader
    """
    if isinstance(index_names, str):
        index_names = list(index_names)
    print(index_names)

    def indexer(path_to_img: PathLike) -> pd.Series:
        inds = re.findall(r"\d+", Path(path_to_img).name)
        series = pd.Series(inds, index=index_names).astype(int)
        return series

    return indexer


def load_tiffs(
    *,
    files: Iterable[PathLike] = None,
    folder: PathLike = None,
    indexer: callable | str = None,
) -> xr.DataArray:
    """
    Load a collection of tiffs that may or may not be complete, dropping any timepoints
    that aren't complete.

    Parameters
    ----------
    files : Iterable[Pathlike]
        An interable of the files to load. Incompatible with *folder* argument
    folder : Pathlike
        The folder containing tiff files. Incompatible argument with *files* argument.
    indexer : Callable, or str, optional
        Indexer to use. If None use indexer for raman-mda-engine. If a str
        then passed on to *indexer_maker*.

    Returns
    -------
    xr.DataArray
        A dask backed data array of the images.
    """
    nones = [files is None, folder is None]
    if all(nones) or not any(nones):
        raise ValueError("Exactly one of *files* and *folder* must not be None.")
    if indexer is None:
        indexer = indexer_maker()
    elif isinstance(indexer, str):
        indexer = indexer_maker(indexer)

    if files is None:
        files = pd.Series(Path(folder).glob("*.tiff"))
        if len(files) == 0:
            # try with single f tiff
            files = pd.Series(Path(folder).glob("*.tif"))
    else:
        files = pd.Series(files)
    if len(files) == 0:
        raise ValueError("No files found")

    index = files.apply(indexer)
    idx = index_complete_timepoints(index)
    index = index[idx]
    files = files[idx]

    # make list to work around https://github.com/AllenCellModeling/aicsimageio/pull/449
    reader = TiffGlobReader([str(f) for f in files], indexer=index)
    return reader.get_xarray_dask_stack()
