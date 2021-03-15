__all__: [
    "manual_segmentation",
    "correct_watershed",
    "correct_decreasing_cell_frames",
]
import numpy as np
import xarray as xr
import warnings
from .array_utils import axis2int
from .segmentation import (
    watershed_single_frame_preseeded,
    peak_mask_to_napari_points,
    napari_points_to_peak_mask,
)
from .track_utils import reindex_labels, find_bad_frames
from skimage.segmentation import relabel_sequential

try:
    import napari
except ImportError:
    napari = None
    warnings.warn(
        "Could not import napari. The function *manual_segmentation* will fail if you call it",
        stacklevel=3,
    )


def scroll_time(viewer, time_axis=1):
    def scroll_callback(layer, event):
        if "Shift" in event.modifiers:
            new = list(viewer.dims.current_step)

            # get the max time
            max_time = viewer.dims.range[time_axis][1]

            # event.delta is (float, float) for horizontal and vertical scroll
            # on linux shift-scroll gives vertical
            # but on mac it gives horizontal. So just take the max and hope
            # for the best
            if max(event.delta) > 0:
                if new[time_axis] < max_time:
                    new[time_axis] += 1
            else:
                if new[time_axis] > 0:
                    new[time_axis] -= 1
            viewer.dims.current_step = new

    viewer.mouse_wheel_callbacks.append(scroll_callback)


def apply_label_keybinds(labels):
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
    time_axis = 1

    def scroll_callback(layer, event):
        if len(event.modifiers) == 0 and labels.mode in ["paint", "erase"]:
            if event.delta[1] > 0:
                labels.brush_size += 1
            else:
                labels.brush_size = max(labels.brush_size - 1, 1)

    labels.mouse_wheel_callbacks.append(scroll_callback)


def apply_points_keybinds(points):
    @points.bind_key('q')
    def remove_selected(layer):
        points.remove_selected()

    @points.bind_key('w')
    def add_mode(layer):
        points.mode = 'add'

    @points.bind_key('e')
    def select_mode(layer):
        points.mode = 'select'


def manual_segmentation(img, mask=None, time_axis='T'):
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
    time_axis : str or int or None, default: 'T'
        Which axis to treat as the time axis for shift-scroll.
        If None or a string when img is an xarray then the first axis will be used.

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

    time_axis = axis2int(img, axis=time_axis, fallthrough=0)
    with napari.gui_qt():
        # create the viewer and add the cells image
        viewer = napari.view_image(img, name="cells")
        # add the labels
        labels = viewer.add_labels(mask, name="segmentation")
        # Add more keybinds for better ergonomics
        apply_label_keybinds(labels)
        scroll_time(viewer)

    if isinstance(img, xr.DataArray):
        return xr.DataArray(labels.data, coords=img.coords, dims=img.dims)
    else:
        return labels.data


def correct_watershed(ds):
    """
    Manually correct parts of an image with a bad watershed.
    This will modify the 'peak_mask' and 'labels' variables of the Dataset inplace.

    Keybindings:

    2 : toggle between mask and labels
    3 : toggle between controlling mask/labels or points
    4 : Toggle visibility of mask+labels on and off
    5 : Toggle whether painting paints through all time points
    Control-l : rereun current frame's watershed
    Shift + Scroll : scrub through time points

    Point Layer Keybindings:
    q : delete selected points
    w : Switch to `Add Points` mode
    e : Switch to `Select` mode

    Mask/Labels layer Keybindngs:
    q : erase
    w : fill
    e : paint
    r : pick
    t : create new label
    s : fill with background

    Scroll : modify brush size when in paint mode

    Parameters
    ----------
    ds : (S, T, ... , Y, X) xarray dataset
    """

    viewer = napari.view_image(ds["images"].sel(C="BF"))
    labels = viewer.add_labels(ds["labels"].values, name="labels", visible=False)
    mask = viewer.add_labels(ds["mask"].values, name="mask", visible=True)
    points = viewer.add_points(peak_mask_to_napari_points(ds['peak_mask']), size=1)
    apply_label_keybinds(labels)
    apply_label_keybinds(mask)
    scroll_time(viewer)
    apply_points_keybinds(points)

    through_time = False

    def setup_through_time(layer):
        old_paint = layer.paint
        old_fill = layer.fill

        def paint_through_time(coord, new_label, refresh=True):
            if through_time:
                for i in range(labels.data.shape[1]):
                    c = list(coord)
                    c[1] = i
                    old_paint(c, new_label, refresh)
            else:
                old_paint(coord, new_label, refresh)

        # doesn't work because napari fill only allows filling the currently
        # viewed slice
        # https://github.com/napari/napari/blob/402771b0a331a62a891fd0a08c2f698424d51633/napari/layers/labels/labels.py#L809-L816
        # def fill_through_time(coord, new_label, refresh=True):
        #     print('here??')
        #     if through_time:
        #         print('here2')
        #         for i in range(labels.data.shape[1]):
        #             c = list(coord)
        #             c[1] = i
        #             print(c)
        #             old_fill(c, new_label, refresh)
        # else:
        #     old_fill(coord, new_label, refresh)

        layer.paint = paint_through_time
        # layer.fill = fill_through_time

    setup_through_time(labels)
    setup_through_time(mask)

    def toggle_masks(*args):
        if mask.visible and labels.visible:
            # ugh - I guess set the labels to visible
            labels_and_points()
        elif mask.visible:
            labels_and_points()
        else:
            mask_and_points()
        if viewer.active_layer in [mask, labels]:
            set_correct_active_labels()

    def mask_and_points(*args):
        mask.visible = True
        labels.visible = False

    def labels_and_points(*args):
        mask.visible = False
        labels.visible = True

    layer_arr = np.array([labels, mask])

    def set_correct_active_labels():
        """If a labels layer is active make sure it is the correct one"""
        viewer.layers.unselect_all()
        new_layer = layer_arr[[labels.visible, mask.visible]][0]
        # using seleccted instead of active layer due to
        # https://github.com/napari/napari/issues/2390
        new_layer.selected = True

    def toggle_points_vs_labels(viewer):
        if viewer.active_layer == points:
            set_correct_active_labels()
        else:
            viewer.layers.unselect_all()
            points.selected = True

    def gogogo(viewer):
        labels_and_points()
        S, T = viewer.dims.current_step[:2]
        ds['peak_mask'][S, T] = napari_points_to_peak_mask(
            points.data, (ds.dims['Y'], ds.dims['X']), S, T
        )
        watershed_single_frame_preseeded(ds, S, T)
        labels.data = ds['labels'].values

    _lastmask = mask

    def toggle_bf_mask(viewer):
        nonlocal _lastmask
        if mask.visible or labels.visible:
            if mask.visible:
                _lastmask = mask
                mask.visible = False
            if labels.visible:
                _lastmask = labels
                labels.visible = False

        else:
            _lastmask.visible = True
            set_correct_active_labels()

    def toggle_through_time(*args):
        nonlocal through_time
        through_time = not through_time

    viewer.bind_key("2", toggle_masks)
    viewer.bind_key("3", toggle_points_vs_labels)
    viewer.bind_key("4", toggle_bf_mask)
    viewer.bind_key("5", toggle_through_time)
    viewer.bind_key("Control-l", gogogo)


def correct_decreasing_cell_frames(ds, bad_frames=None, extra_labels=None):
    """
    Show only the pairs of frames for which cell number decreasing.
    This will modify *ds['labels']* in place when closed or when `ctrl-shift-d` pressed.

    Controls:
    Labels editing the same as always
    Control-Shift-d : check the values and change what is displayed to the problem frames.

    Parameters
    ----------
    ds : (S, T, ..., Y, X) Dataset
    bad_frames : list of tuple of int, optional
        If *None*, then `find_bad_frames` will be used
    extra_labels : str or list of strings
        Other channels in the dataset to view. Will be added
        as a napari label layer so should probably be binary images.

    Returns
    -------
    viewer : Napari viewer object
    """
    if extra_labels is not None:
        if isinstance(extra_labels, str):
            extra_labels = [extra_labels]
    else:
        extra_labels = []

    def gen_data(bad_frames=None):
        if bad_frames is None:
            bad_frames = find_bad_frames(ds)
        s_idx = []
        t_idx = []
        for i in bad_frames:
            s_idx.extend([i[0], i[0]])
            t_idx.extend([i[1] - 1, i[1]])
        BF = (
            ds['images']
            .sel(C='BF')
            .values[:][tuple(s_idx), tuple(t_idx)]
            .reshape(len(t_idx) // 2, 2, *ds['labels'].shape[-2:])
        )
        indiv = (
            ds['labels']
            .values[:][tuple(s_idx), tuple(t_idx)]
            .reshape(len(t_idx) // 2, 2, *ds['labels'].shape[-2:])
        )

        other_layers = [
            (
                ds[other]
                .values[:][tuple(s_idx), tuple(t_idx)]
                .reshape(len(t_idx) // 2, 2, *ds[other].shape[-2:])
            )
            for other in extra_labels
        ]

        return BF, indiv, other_layers, s_idx, t_idx

    def reassign():
        """
        Because the reshape of the original values makes a copy rather than a view :(
        so standard in place editing doesn't work
        """
        indiv = ds['labels'].values[:][tuple(s_idx), tuple(t_idx)] = labels.data.reshape(
            len(t_idx), *labels.data.shape[-2:]
        )

    def check_all(*args):
        reassign()
        nonlocal t_idx, s_idx
        BF, indiv, other_layers, s_idx, t_idx = gen_data(None)
        image.data = BF
        for data, layer in zip(other_layers, others):
            layer.data = data

        labels.data = indiv

    BF, indiv, other_layers, s_idx, t_idx = gen_data(bad_frames)
    viewer = napari.Viewer()
    image = viewer.add_image(BF)
    labels = viewer.add_labels(indiv)
    others = [viewer.add_labels(other) for other in other_layers]
    viewer.unselect_all()
    label.selected = True

    apply_label_keybinds(labels)
    scroll_time(viewer)
    viewer.bind_key('Control-Shift-d', check_all)

    def on_close(*args, **kwargs):
        reassign()

    # this on_close may not work in the future. See discussion on zulip
    # https://napari.zulipchat.com/#narrow/stream/212875-general/topic/on-close/near/230088585
    viewer.window._qt_window.destroyed.connect(on_close)

    return viewer
