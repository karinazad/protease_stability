import pandas as pd
import numpy as np
from src_.config import Config
from src_.utils.general import tokenize_and_pad_sequences, drop_nans, add_start_end_characters
from src_.utils.general import multi_target_train_test_split


def get_and_process_data(data_path, return_as_df=False, clip=True):
    range_kT = (-6, 2)
    range_kC = (-6, 0.5)

    data = pd.read_csv(data_path)
    seq_aa = data.aa_seq
    kT, kC = data.k_T_1, data.k_C_1

    if clip:
        kT = np.clip(kT, range_kT[0], range_kT[1])
        kC = np.clip(kC, range_kC[0], range_kC[1])

    # Remove rows where the amino acid sequence is missing
    X, [kT, kC] = drop_nans(X=seq_aa, targets=[kT, kC])

    # Add start ("J") and end ("O") characters
    X_ = X.apply(add_start_end_characters)

    # Tokenize letters to integers and pad sequences to the maximum length
    X_ = tokenize_and_pad_sequences(X_, num_words=Config.get("n_char"), max_len=Config.get("seq_length"))

    if return_as_df:
        X = pd.DataFrame(X_, index=X.index)
    else:
        X = X_

    return X, kT, kC


def get_folded_unfolded_data_splits(
        X_unfolded,
        kT_unfolded,
        kC_unfolded,
        X_folded,
        kT_folded,
        kC_folded,
):
    np.random.seed(0)

    # Train test split for unfolded
    X_unfolded_train, X_unfolded_test, kT_unfolded_train, kT_unfolded_test, kC_unfolded_train, kC_unfolded_test = \
        multi_target_train_test_split(X_unfolded, kT_unfolded, kC_unfolded, return_val=False)

    # Train test split for folded
    # To have balanced training size, select the same number of folded samples as unfolded

    indices = np.random.randint(low=0, high=X_folded.shape[0], size=(X_unfolded_train.shape[0],))

    X_folded_train, kT_folded_train, kC_folded_train = \
        list(map(lambda x: x[indices], [X_folded, kT_folded, kC_folded]))

    assert X_unfolded_train.shape == X_folded_train.shape
    assert kT_unfolded_train.shape == kT_folded_train.shape
    assert kC_unfolded_train.shape == kC_folded_train.shape

    # Select those samples that were not included in the training data
    mask = np.ones(kT_folded.shape, bool)
    mask[indices] = False

    X_folded_test, kT_folded_test, kC_folded_test = \
        list(map(lambda x: x[mask], [X_folded, kT_folded, kC_folded]))

    unfolded_data = {
        "X_train": X_unfolded_train,
        "X_test": X_unfolded_test,
        "kT_train": kT_unfolded_train,
        "kT_test": kT_unfolded_test,
        "kC_train": kC_unfolded_train,
        "kC_test": kC_unfolded_test,
    }

    folded_data = {
        "X_train": X_folded_train,
        "X_test": X_folded_test,
        "kT_train": kT_folded_train,
        "kT_test": kT_folded_test,
        "kC_train": kC_folded_train,
        "kC_test": kC_folded_test,
    }
    return unfolded_data, folded_data
