__all__ = [
    "setup_zarrs",
]


def setup_zarrs(root, group_name, sizes):
    """
    Create a group in root and set up arrays to hold the outputs of cellpose
    (masks, probabilities, and flows).
    """

    group = root.create_group(group_name)

    _ = group.zeros(
        'masks', shape=(sizes['T'], sizes['Y'], sizes['X']), dtype='uint16', chunks=(1, -1, -1)
    )
    _ = group.zeros('probs', shape=(sizes['T'], sizes['Y'], sizes['X']), chunks=(1, -1, -1))
    _ = group.zeros('flows', shape=(sizes['T'], 2, sizes['Y'], sizes['X']))

    return group
