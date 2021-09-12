import pandas as pd
from src_.config import Config
from src_.utils.general import tokenize_and_pad_sequences, drop_nans, add_start_end_characters


def get_and_process_data(data_path, return_as_df=False):
    data = pd.read_csv(data_path)
    seq_aa = data.aa_seq
    kT, kC = data.k_T_1, data.k_C_1

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


