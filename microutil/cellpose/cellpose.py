__all__ = [
    "setup_cellpose_zarrs",
]


def setup_cellpose_zarrs(group, img_shape):
    """
    Create a group in root and set up arrays to hold the outputs of cellpose
    (masks, probabilities, and flows).
    """
    _ = group.zeros('masks', shape=img_shape, dtype='uint16', chunks=(1, -1, -1))
    _ = group.zeros('flows', shape=(img_shape[0], 2, *img_shape[1:]), chunks=(1, -1, -1))
    _ = group.zeros('probs', shape=img_shape, chunks=(1, -1, -1))
    _ = group.zeros('styles', shape=(img_shape[0], 256), chunks=(1, -1))

    return group
