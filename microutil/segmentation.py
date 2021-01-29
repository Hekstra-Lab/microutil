__all__ = [
    "unet",
    "manual_segmentation",
]

from ._unet import unet
import napari
import numpy as np
import xarray as xr
import dask.array as da


def manual_segmentation(img, mask=None):
    """
    Open up Napari

    Parameters
    ----------
    img : image-like
        Last to dims should be XY. You probably want this to be a BF image.
    mask : array-like or None
        If array-like it should be broadcastable to the same dims as *img*

    Returns
    -------
    mask :
        The mask that was updated by user interactions
    """
    if mask is None:
        if isinstance(img, np.ndarray):
            mask = np.zeros_like(img)
        elif isinstance(img, xr.DataArray):
            mask = xr.zeros_like(img)
        elif isinstance(img, da.Array):
            mask = np.zeros_like(img)
    

    with napari.gui_qt():
        # create the viewer and add the cells image
        viewer = napari.view_image(img, name="cells")
        # add the labels
        labels = viewer.add_labels(mask, name="segmentation")
        # Add more keybinds for better ergonomics
        @labels.bind_key("q")
        def paint_mode(viewer):
            labels.mode = "erase"

        @labels.bind_key("w")
        def paint_mode(viewer):
            labels.mode = "fill"

        @labels.bind_key("e")
        def paint_mode(viewer):
            labels.mode = "paint"

        @labels.bind_key("r")
        def paint_mode(viewer):
            labels.mode = "pick"

        @labels.bind_key("t")
        def new_cell(viewer):
            labels.selected_label = labels.data.max() + 1

            
        # scrolling in paint mode changes the brush size
        def brush_size_callback(layer, event):
            if labels.mode in ["paint", "erase"]:
                if event.delta[1] > 0:
                    labels.brush_size += 1
                else:
                    labels.brush_size = max(labels.brush_size-1,1)

        labels.mouse_wheel_callbacks.append(brush_size_callback)


    return labels.data
