import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.stats import linregress
from skimage.filters import sobel
import scipy.ndimage as ndi

__all__ = [
    "angular_offset",
    "rotate",
    "crop",
    "pad_find_peaks",
    "trench_ripper",
    "fine_trench_rip",
    "gogogo_trapathon",
]


def angular_offset_single(im, width=70, prominence=0.3, edges=None, edge_sum_thresh=10):
    """
    Calculate a global angular offset for a single FOV based on computed edges
    in the BF. Inteded to be called by xarray.apply_func in `angular_offset`.
    Default values should work for 40x images from Harvard CBS Nikon TE2000.


    Parameters
    ----------
    im : np.array
      Array containing a single image. Assumes the traps are oriented vertically (along the 0 axis)
      and that the flow channels are horizontal (axis=1)
    width: int, default 70
      Approximate vertical width of trapping region.     prominence: float, default 0.3
      Prominence of peaks passed to `find_peaks` for identifying top and bottom of the trapping region.
    edges: None or np.array shame shape as im
      If you have precomputed the edges, you can pass them in directly.
    edge_sum_thresh: float, default 10
      Threshold for ruling out parts of image that contain traps.

    Returns
    -------
    rotation: float
      Angle in radians by which im could be rotated in order to orient the trapps to be vertical.
    """

    if edges is None:
        edges = sobel(im)

    xx = np.arange(edges.shape[1])
    not_traps = edges[:, edges.sum(0) < edge_sum_thresh]
    xx = xx[edges.sum(0) < edge_sum_thresh]

    use_idx = np.random.choice(not_traps.shape[1], size=300, replace=False)
    xx = xx[use_idx]
    use_edges = not_traps[:, use_idx]

    top_bottom = [find_peaks(e, prominence=prominence)[0] for e in use_edges.T]
    pairs = []
    x_pairs = []
    for i, x in enumerate(top_bottom):
        if len(x) == 2 and x[1] - x[0] > width:
            pairs.append(x)
            x_pairs.append(xx[i])
    pairs = np.array(pairs)
    x_pairs = np.array(x_pairs)

    top_slope = linregress(x_pairs, pairs[:, 0]).slope
    bottom_slope = linregress(x_pairs, pairs[:, 1]).slope

    rotation = np.arctan(0.5 * (top_slope + bottom_slope))
    return rotation


def angular_offset(bf, dims='STCZYX', **kwargs):
    """
    Calculate the angular offset required to align each image such that the traps are oriented vertically.
    Wraps `angular_offset_single` with `xarray.apply_ufunc`. kwargs get passed directly to `angular_offset_single`.

    Parameters
    ----------
    bf: xarray.DataArray
      DataArray containing brightfield images.
    dims: str or list of str, default 'STCZYX`
      Dimensions names for `bf` that correspond to STCZYX

    Returns
    -------
    angles: xr.DataArray with all but X,Y dimensions of bf
      Angle to rotate each frame by to render the traps vertically in the image.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    return xr.apply_ufunc(
        angular_offset_single,
        bf,
        input_core_dims=[[Y, X]],
        vectorize=True,
        dask='parallelized',
        kwargs=kwargs,
        output_dtypes=['float64'],
    )


def rotate(ds, angles, dims='STCZYX', **kwargs):
    """
    Wrap scipy.ndimage.rotate to vectorize over frames of xr.Datset.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    degrees = np.rad2deg(angles)

    return xr.apply_ufunc(
        ndi.rotate,
        ds,
        degrees,
        input_core_dims=[[Y, X], []],
        output_core_dims=[[Y, X]],
        vectorize=True,
        dask='parallelized',
        kwargs={'reshape': False},
        output_dtypes=['uint16'],
    )


def crop(ds, angles, dims='STCYZX'):
    """
    Parameters
    ----------
    ds : xr.Dataset
        Datset containing image data to be cropped.
    angles : xr.DataArray
        Rotation angle which will be accounted for in crop.
    dims: str or list of str, default 'STCZYX`
        Dimensions names for `bf` that correspond to STCZYX

    Returns
    -------
    cropped : xr.Dataset
        Dataset cropped to remove edge effects of rotating by angle.
    """
    slopes = np.tan(angles)
    crops_y = np.ceil(np.max(ds.sizes['Y'] * np.abs(slopes))).astype(int).item()
    crops_x = np.ceil(np.max(ds.sizes['X'] * np.abs(slopes))).astype(int).item()
    return ds.isel(Y=slice(crops_y, -crops_y), X=slice(crops_x, -crops_x))


def pad_find_peaks(trace, distance, prominence, pad=40):
    """
    Wrap scipy.signal.find_peaks to return peak locations padded with -1
    in order to be used in arrays.

    Parameters
    ----------

    trace : 1D array like
       Signal in which to locate peaks
    distance : float
        Minimum distance between peaks. Passed directly to find_peaks
    prominence : float
        Minimum prominence of each peak. Passed directly to find_peaks.
    pad : int default 40
        Number of elements in final array.

    Returns
    -------
    padded : 1D array like
        Array with peak locations padded on the right with -1s.
    """
    peaks = find_peaks(trace.copy(), distance=distance, prominence=prominence)[0]
    padded = np.pad(peaks, (0, pad - peaks.shape[0]), constant_values=-1)
    return padded


def trench_ripper(
    ds,
    out_width=60,
    out_height=150,
    trap_width=10,
    prominence=0.017,
    edges=None,
    image_name='images',
    bf_name='BF',
    dims='STCZYX',
):
    """
    Take in full fovs. Identify trenches and split them out.
    Then center each trench in its frame and crop.


    Parameters
    ----------
    ds : xr.Dataset
       Dataset containing at least brightfield images.
    out_width : int, default 60
        Width of final, single trap images in pixels
    out_height : int, default 150
        Height of final single trap images in pixels
    trap_width : int default 10
        Approximate width of an individual trap in pixels.
    prominence : float default 0.017
        Prominence of peaks in image edges. Passed to find_peaks_cwt
    edges : None or DataArray
        If None, edges will be computed with a sobel filter. If edges have been
        pre computed they can be passed here.
    image_name : str default 'images'
        Variable name in ds that corresponds to image data.
    bf_name : str default 'BF'
        Coordinate value within the images data variable that corresponds to BF images.
    dims: str or list of str, default 'STCZYX`
      Dimensions names for `bf` that correspond to STCZYX

    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    else:
        S, T, C, Z, Y, X = dims

    if edges is None:
        # print('calculating edges')
        edges = xr.apply_ufunc(
            sobel,
            ds[image_name].sel({C: bf_name}),
            input_core_dims=[[Y, X]],
            output_core_dims=[[Y, X]],
            vectorize=True,
            dask='parallelized',
        )

    ds['edges'] = edges.load()

    trap_edges = xr.apply_ufunc(
        pad_find_peaks,
        edges.mean(Y).mean(T),
        kwargs={'distance': trap_width, 'prominence': prominence},
        input_core_dims=[[X]],
        output_core_dims=[[X]],
        exclude_dims=set((X,)),
        vectorize=True,
        dask='parallelized',
    )
    # print(trap_edges)
    # rough pass
    rough_trench_view = []
    # loop through positions
    for i in range(ds.sizes[S]):

        x = trap_edges.isel(S=i)
        trenches = []
        diff = np.diff(x[x >= 0])
        left_sides = np.argwhere(diff < out_width / 2).squeeze().astype(int)
        # loop over trenches
        for l, k in enumerate(left_sides):
            # by_time.append([])
            dx = x[k + 1] - x[k]
            pad_width = 1.25 * out_width - dx
            left = x[k] - pad_width / 2
            right = x[k + 1] + pad_width / 2
            if left < 0:
                right -= left
                left = 0
            if right > ds.sizes[X]:
                left -= right - ds.sizes[X]
                right = ds.sizes[X]
            rough_trench_view.append(ds.isel({S: i, X: slice(int(left), int(right))}))
        # print(f"trench ripping S={i}-- found {len(rough_trench_view)} trenches")
    return xr.concat(rough_trench_view, dim='Trench')


def fine_trench_rip(
    tr,
    out_width=60,
    out_height=150,
    trap_width=10,
    prominence=0.017,
    edges=None,
    image_name='images',
    bf_name='BF',
    dims='STCZYX',
):
    """
    Second pass over the ripped trenches to more carefully center the trenches in the field.

    Parameters
    ----------
    tr : xr.Dataset
       Dataset containing views of individual trenches. Should be the output of trench_ripper.
    out_width : int, default 60
        Width of final, single trap images in pixels
    out_height : int, default 150
        Height of final single trap images in pixels
    trap_width : int default 10
        Approximate width of an individual trap in pixels.
    prominence : float default 0.017
        Prominence of peaks in image edges. Passed to find_peaks_cwt
    edges : None or DataArray
        If None, edges will be computed with a sobel filter. If edges have been
        pre computed they can be passed here.
    image_name : str default 'images'
        Variable name in ds that corresponds to image data.
    bf_name : str default 'BF'
        Coordinate value within the images data variable that corresponds to BF images.
    dims: str or list of str, default 'STCZYX`
      Dimensions names for `bf` that correspond to STCZYX
    
    Returns
    -------
    Y_lims, X_lims : DataArray
        Y and X limits of each frame to center the trench in the frame.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    else:
        S, T, C, Z, Y, X = dims

    py = 10 * prominence * np.ones(tr.edges.sizes[Y])  # why 10x? No idea
    py[: int(tr.edges.sizes[Y] / 10)] = 10
    py[-int(tr.edges.sizes[Y] / 10) :] = 10

    px = prominence * np.ones(tr.edges.sizes[X])
    px[: int(tr.edges.sizes[X] / 10)] = 10
    px[-int(tr.edges.sizes[X] / 10) :] = 10
    sizes = tr.images.sizes

    def indiv_crop(
        edges, out_height, out_width, trap_width, sizes, px, py, dims=['Trench'] + list('CTZYX')
    ):
        if isinstance(dims, str):
            S, T, C, Z, Y, X = list(dims)
        else:
            S, T, C, Z, Y, X = dims
        ypeaks = pad_find_peaks(edges.mean(1), out_height / 3, py, 2)
        try:
            y1, y2 = ypeaks  # .isel(Trench=i, T=t).values
        except:
            y1 = int((sizes[Y] - out_height) / 2)
            y2 = sizes[Y] - y1
        dy = y2 - y1
        pad_height = out_height - dy
        top = y1 - pad_height / 2
        bottom = y2 + pad_height / 2
        if top < 0:
            bottom -= top
            top = 0
        if bottom > sizes[Y]:
            top -= bottom - trench.sizes[Y]
            bottom = sizes[X]

        xpeaks = pad_find_peaks(edges.mean(0), trap_width, px, 2)
        try:
            x1, x2 = xpeaks  # .isel(Trench=i, T=t).values
        except:
            x1 = int((sizes[X] - out_width) / 2)
            x2 = sizes[Y] - x1
        dx = x2 - x1
        pad_width = out_width - dx
        left = x1 - pad_width / 2
        right = x2 + pad_width / 2
        if left < 0:
            right -= left
            left = 0
        if right > sizes[X]:
            left -= right - sizes[X]
            right = sizes[X]
        return xr.DataArray(np.arange(int(top), int(bottom)), dims=[Y]), xr.DataArray(
            np.arange(int(left), int(right)), dims=[X]
        )

    final = xr.apply_ufunc(
        indiv_crop,
        tr.edges.transpose(..., Y, X),
        input_core_dims=[[Y, X]],
        kwargs={
            'px': px,
            'py': py,
            'sizes': sizes,
            'out_height': out_height,
            'out_width': out_width,
            'trap_width': trap_width,
        },
        output_core_dims=[[Y], [X]],
        exclude_dims=set([Y, X]),
        vectorize=True,
        dask='parallelized',
    )

    return final


def gogogo_trapathon(
    ds,
    out_width=60,
    out_height=150,
    trap_width=10,
    prominence=0.017,
    edges=None,
    image_name='images',
    bf_name='BF',
    load_vars=[],
    dims='STCZYX',
):
    """
    Parameters
    ----------
    ds : xr.Dataset
       Dataset containing at least brightfield images.
    out_width : int, default 60
        Width of final, single trap images in pixels
    out_height : int, default 150
        Height of final single trap images in pixels
    trap_width : int default 10
        Approximate width of an individual trap in pixels.
    prominence : float default 0.017
        Prominence of peaks in image edges. Passed to find_peaks_cwt
    edges : None or DataArray
        If None, edges will be computed with a sobel filter. If edges have been
        pre computed they can be passed here.
    image_name : str default 'images'
        Variable name in ds that corresponds to image data.
    bf_name : str default 'BF'
        Coordinate value within the images data variable that corresponds to BF images.
    dims: str or list of str, default 'STCZYX`
      Dimensions names for `bf` that correspond to STCZYX

    Returns
    -------
    trench_vew: xr.Dataset
        Dataset with image variable
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    else:
        S, T, C, Z, Y, X = dims
    # compute position wise angular offsets
    angles = angular_offset(ds[image_name].sel({C: bf_name}), dims=dims).load()
    rotated = rotate(ds, angles, dims=dims)
    cropped = crop(rotated, angles, dims=dims)
    #    print(cropped.sizes)
    #    print(edges)
    print("Done with global alignment")
    rough = trench_ripper(
        cropped, out_width, out_height, trap_width, prominence, edges, image_name, bf_name, dims
    )
    print("Done ripping trenches")

    y_idx, x_idx = fine_trench_rip(
        rough, out_width, out_height, trap_width, prominence, edges, image_name, bf_name, dims
    )
    print("computed second pass inds")

    load_vars += [image_name, 'edges']

    rough = rough[load_vars]
    rough.load()
    print("Loaded necessary data")
    return rough.isel(Y=y_idx, X=x_idx)


### TODO make function to screen for shitty trenches
