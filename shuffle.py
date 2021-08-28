import numpy as np

def shuffle_arrays(arrays, set_seed=None):
    # copied from https://stackoverflow.com/a/51526109
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(
        0, 2**(32 - 1) - 1) if set_seed is None else set_seed
    for arr in arrays:
        shuffle_array(arr, set_seed=seed)


def shuffle_array(arr, set_seed=None):
    rstate = np.random.RandomState(set_seed)
    rstate.shuffle(arr)

def random_choice(arr, n_sample, set_seed=None):
    assert arr.shape[0] >= n_sample
    seed = np.random.randint(
        0, 2**(32 - 1) - 1) if set_seed is None else set_seed
    rstate = np.random.RandomState(seed)
    length = len(arr)
    return arr[rstate.choice(length, n_sample, replace=False)]