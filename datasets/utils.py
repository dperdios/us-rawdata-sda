import os
import us.utils.picmus as picmus
import numpy as np
import h5py
from typing import List, Tuple, Union
from us.sequence import PWSequence


def load_ius2017_train_set() -> np.ndarray:
    path_train_list = []
    base_path = os.path.join('datasets', 'L11-4v_train')
    path_train_list.append(os.path.join(base_path, 'L11-4v_train_00000.hdf5'))
    path_train_list.append(os.path.join(base_path, 'L11-4v_train_00001.hdf5'))
    train_set = _load_train_list(path_list=path_train_list)
    # Reshape
    return train_set.reshape((train_set.shape[0] * train_set.shape[1], train_set.shape[2]))


def load_ius2017_test_set(picmus16: bool = True, picmus17: bool = True) -> List[PWSequence]:
    # PICMUS set
    test_path_p16 = os.path.join('datasets', 'picmus16')
    test_path_p17 = os.path.join('datasets', 'picmus17')

    if picmus16:
        test_set_p16 = _load_picmus_dir(test_path_p16)
    else:
        test_set_p16 = []
    if picmus17:
        test_set_p17 = _load_picmus_dir(test_path_p17)
    else:
        test_set_p17 = []

    return test_set_p16 + test_set_p17


def generate_cross_valid_sets(full_set: np.ndarray, valid_size: int = None,
                              seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    prng = np.random.RandomState(seed=seed)
    prng.shuffle(full_set)
    # Split in valid_set and train_set
    valid_set, train_set = np.split(full_set, [valid_size])
    return valid_set, train_set


def _load_train_list(path_list: List[str]) -> np.ndarray:
    # Check total size of
    shapes = []
    dtypes = []
    for path in path_list:
        with h5py.File(path, 'r') as h5file:
            if not list(h5file)[0] == 'DATA':
                raise TypeError('Unsupported dataset')
            shapes.append(h5file['DATA'].shape)
            dtypes.append(h5file['DATA'].dtype)

    # Load
    #   Dimensions 1 and 2 must all be the same
    dim0 = [s[0] for s in shapes]
    dim12 = [(s[1], s[2]) for s in shapes]
    if not dim12.count(dim12[0]) == len(dim12):
        raise ValueError('Dimensions 1 and 2 must be the same')
    #   dtypes must be the same
    if not dtypes.count(dtypes[0]) == len(dtypes):
        raise TypeError('dtypes must be the same')
    dtype = dtypes[0]
    arr_shape = list(shapes[0])  # shapes[0] is a tuple, hence needs to be converted
    arr_shape[0] = np.sum(dim0)
    #   Allocate memory
    train_set = np.zeros(arr_shape, dtype=dtype)
    #   Loop in the files
    ind_start = 0
    for path in path_list:
        with h5py.File(path, 'r') as h5file:
            if not list(h5file)[0] == 'DATA':
                raise TypeError('Unsupported dataset')
            ind_end = ind_start + h5file['DATA'].shape[0]
            train_set[ind_start:ind_end] = h5file['DATA'].value
            # Update index
            ind_start = ind_end

    return train_set


def _load_picmus_dir(path: str) -> List[PWSequence]:

    sample_number = 1024
    seq_list = []

    file_list = os.listdir(path)
    file_list.sort()
    for filename in file_list:
        if 'numerical' in filename:
            remove_tgc = False
            first_index = 300
            selected_index = None
        elif 'in_vitro_type1' in filename:
            remove_tgc = True
            first_index = 275
            selected_index = None
        elif 'in_vitro_type2' in filename:
            remove_tgc = True
            first_index = 375
            selected_index = None
        elif 'in_vitro_type3' in filename:
            remove_tgc = True
            first_index = 400
            selected_index = None
        elif 'carotid_cross_expe' in filename:
            remove_tgc = False
            first_index = 150
            selected_index = [37]
        elif 'carotid_long_expe' in filename:
            remove_tgc = False
            first_index = 100
            selected_index = [37]
        else:
            raise NameError('Unsupported PICMUS data')

        # Load PWSequence
        seq_path = os.path.join(path, filename)
        seq = picmus.import_sequence(path=seq_path, remove_tgc=remove_tgc, selected_indexes=selected_index)

        # Crop data
        seq.crop_data(first_index=first_index, last_index=first_index + sample_number)

        # Apply exponential TGC
        if remove_tgc:
            seq.apply_tgc()

        # Normalize data
        seq.normalize_data()

        # Add to list
        seq_list.append(seq)

    return seq_list


