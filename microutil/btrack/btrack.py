import btrack
import numpy as np
from skimage.util import map_array

__all__ = [
    "gogogo_btrack",
]


def gogogo_btrack(
    labels,
    config_file,
    radius,
    tracks_out=None,
    intensity_image=None,
    properties=(),
    use_weighted_centroid=True,
):
    """
    Run btrack on a single position timeseries. Write the track data to h5
    files using the build in export capability of btrack.

    Parameters
    ----------
    labels : np.ndarray
        Images containing single cell labels
    config_file : str
        Path to btrack config file.
    radius : int
        Maximum search radius for btrack
    tracks_out : str
        Path to h5 files where track data will be saved
    intensity_image: np.ndarray or None
        Underlying intensity image from which to compute optional features
    properties : tuple[str,...]
        Other properties to compute
    use_weighted_centroid : bool default True
        Whether to use the weighted centroid for objects if intensity_image is provided.
        Default True.

    Returns
    -------
    updated_labels: np.ndarray
        Array with same shape as labels but with labelled regions
        consistently labelled through time
    """

    objects = btrack.utils.segmentation_to_objects(
        labels,
        intensity_image=intensity_image,
        use_weighted_centroid=use_weighted_centroid,
        properties=properties,
        num_workers=4,
    )

    tracking_updates = ['motion']
    if len(properties) > 0:
        tracking_updates.append('visual')

    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure_from_file(config_file)
        tracker.max_search_radius = radius

        # append the objects to be tracked
        tracker.append(objects)

        # set the tracking volume
        tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))

        # track them (in interactive mode)
        tracker.track(tracking_updates=tracking_updates, step_size=50)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        if tracks_out is not None:
            tracker.export(tracks_out, obj_type='obj_type_1')

        tracks = tracker.tracks

    tracked_labels = btrack.utils.update_segmentation(labels, tracks)
    return tracked_labels


def _tracks_to_labels(segmentation, tracks):
    """
    Map btrack output tracks back into a masked array.

    Parameters
    ----------
    segmentation : np.array
        Array containing a timeseries of single cell masks (dimensions TYX)
    tracks : pd.DataFrame
        btrack output converted to a dataframe

    Returns
    -------
    relabeled : np.array
        Array containing the same masks as segmentation but relabeled to
        maintain single cell identity over time.
    """
    raise DeprecationWarning(
        "tracks_to_labels is deprecated. Prefer btrack.utils.update_segmentation"
    )
    track_positions = tracks.loc[~tracks.dummy, ['ID', 't', 'y', 'x']]
    relabeled = np.zeros_like(segmentation)
    for t, df in track_positions.groupby('t'):
        single_segmentation = segmentation[t]
        new_id, tc, yc, xc = tuple(np.round(df.values).astype(int).T)
        old_id = single_segmentation[yc, xc]
        relabeled[t] = (single_segmentation > 0) * map_array(single_segmentation, old_id, new_id)
    return relabeled
