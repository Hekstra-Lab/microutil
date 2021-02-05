__all__ = [
    "roi_to_mask",
    "process_oval",
]
from read_roi import read_roi_zip
from skimage import draw


def process_oval(info, arr):
    """
    Draw an oval onto *arr* based on the information in *info*.
    This will modify arr in place

    Parameters
    ----------
    info : dict
        Should have keys *width*, *height*, *left*, *top*
    arr : (M, N) array-like
    """
    r_radius = info["width"] / 2
    c_radius = info["height"] / 2
    r = info["left"] + r_radius
    c = info["top"] + c_radius
    rr, cc = draw.ellipse(r, c, r_radius, c_radius, shape=arr.shape)
    arr[cc, rr] = True


def roi_to_mask(roi, shape):
    """
    Convert a micromanager roi to a numpy mask.

    Parameters
    ----------
    roi : str or dict
        The str to the roi zip folder or an already loaded roi zip.
    shape : tuple of int
        The shape of the mask to fill

    Returns
    -------
    mask : numpy array
    """
    if isinstance(roi, str):
        roi = read_roi_zip(roi)

    mask = np.zeros(shape, dtype=bool)
    for info in roi.values():
        if info["type"] == "oval":
            process_oval(info, mask)
    return mask
