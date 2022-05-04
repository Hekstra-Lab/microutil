import btrack
import numpy as np
import pandas as pd
from skimage.util import map_array

__all__ = [
    "gogogo_btrack",
    "tracks_to_labels",
]


def gogogo_btrack(labels, config_file, radius, tracks_out):
    """
    Run btrack on a single position timeseries. Write the track data to h5
    files using the build in export capability of btrack.

    Returns
    -------
    all_tracks : pd.DataFrame
        Dataframe containing track info
    """
    objects = btrack.utils.segmentation_to_objects(labels)

    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure_from_file(config_file)
        tracker.max_search_radius = radius

        # append the objects to be tracked
        tracker.append(objects)

        # set the tracking volume
        tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.export(tracks_out, obj_type='obj_type_1')
    all_tracks = pd.concat([pd.DataFrame(t.to_dict()) for t in tracker.tracks])
    return all_tracks


def tracks_to_labels(segmentation, tracks):
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
    track_positions = tracks.loc[~tracks.dummy, ['ID', 't', 'y', 'x']]
    relabeled = np.zeros_like(segmentation)
    for t, df in track_positions.groupby('t'):
        single_segmentation = segmentation[t]
        new_id, tc, yc, xc = tuple(np.round(df.values).astype(int).T)
        old_id = single_segmentation[yc, xc]
        relabeled[t] = (single_segmentation > 0) * map_array(single_segmentation, old_id, new_id)
    return relabeled
