__all__: [
    "manual_segmentation",
]
import numpy as np
import xarray as xr

try:
    import napari
except ImportError:
    napari = None
    warnings.warn(
        "Could not import napari. The function *manual_segmentation* will fail if you call it",
        stacklevel=3,
    )


def manual_segmentation(img, mask=None):
    """
    Open up Napari set up for manual segmentation. Adds these custom keybindings:
    q : erase
    w : fill
    e : paint
    r : pick
    t : create new label

    s : fill with background

    scroll : modify brush size when in paint mode
    shift + Scroll : scrub through time points

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
    if napari is None:
        raise ImportError("You must install Napari in order to use this function.")
    if mask is None:
        # needs to be numpy as all other options do not seem to work
        # see https://github.com/napari/napari/issues/2190
        mask = np.zeros_like(img, dtype=np.int)
        # if isinstance(img, np.ndarray):
        #     mask = np.zeros_like(img,dtype=np.int)
        # elif isinstance(img, xr.DataArray):
        #     mask = nr.zeros_like(img, dtype=np.int)
        # elif isinstance(img, da.Array):
        #     mask = np.zeros_like(img)
    elif not isinstance(mask, np.ndarray):
        print("casting mask to numpy array")
        print("see https://github.com/napari/napari/issues/2190 for details")
        mask = np.array(mask)

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

        @labels.bind_key("s")
        def paint_mode(viewer):
            labels.selected_label = 0
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
        # shift-scroll changes the time point
        def scroll_callback(layer, event):
            if 'Shift' in event.modifiers:
                new = list(viewer.dims.current_step)
                if event.delta[1] > 0:
                    if new[0] < mask.shape[0] - 1:
                        new[0] += 1
                else:
                    if new[0] > 0:
                        new[0] -= 1
                viewer.dims.current_step = new

            elif labels.mode in ["paint", "erase"]:
                if event.delta[1] > 0:
                    labels.brush_size += 1
                else:
                    labels.brush_size = max(labels.brush_size - 1, 1)

        labels.mouse_wheel_callbacks.append(scroll_callback)

    if isinstance(img, xr.DataArray):
        return xr.DataArray(labels.data, coords=img.coords, dims=img.dims)
    else:
        return labels.data
