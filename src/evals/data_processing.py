import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.utils.general import standard_scale, tokenize_and_pad_sequences, drop_nans


def get_and_process_data(data_path):
    data = pd.read_csv(data_path)
    seq_aa = data.aa_seq
    kT, kC = data.k_T_1, data.k_C_1

    X, [kT, kC] = drop_nans(X=seq_aa, targets=[kT, kC])

    X = tokenize_and_pad_sequences(X,
                                   num_words=Config.config("n_char"),
                                   max_len=Config.config("seq_length"))

    return X, kT, kC


