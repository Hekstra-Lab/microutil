__all__ = [
    "unet",
    "manual_segmentation",
]

from ._unet import unet
import napari


def manual_segmentation(img, mask):
    """
    Open up Napari

    Parameters
    ----------
    img : nD-array
        Last to dims should be XY. You probably want this to be a BF image.
    mask : mD-array

    Returns
    -------
    mask :
        The mask that was updated by user interactions
    """
    with napari.gui_qt():
        # create the viewer and add the cells image
        viewer = napari.view_image(img, name="cells")
        # add the labels
        labels = viewer.add_labels(mask, name="segmentation")
        # Add more keybinds for better ergonomics
        @viewer.bind_key("a")
        def paint_mode(viewer):
            labels.mode = "paint"

        @viewer.bind_key("w")
        def paint_mode(viewer):
            labels.mode = "pick"

        @viewer.bind_key("n")
        def new_cell(viewer):
            labels.selected_label = labels.data.max() + 1

        # scrolling in paint mode changes the brush size
        def brush_size_callback(layer, event):
            if labels.mode == "paint":
                if event.delta[1] > 0:
                    labels.brush_size *= 1.1
                else:
                    labels.brush_size /= 1.1
            labels.mouse_wheel_callbacks.append(brush_size_callback)

    return labels.data