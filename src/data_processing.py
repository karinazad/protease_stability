from src.utils.utils import standard_scale, tokenize_and_pad_sequences
import pandas as pd
from src.globals import N_CHAR, SEQ_LENGTH


def get_and_process_data(data_path):
    data = pd.read_csv(data_path)
    seq_aa = data.aa_seq
    target_t = data.k_T_1
    target_k = data.k_C_1

    target_k_stand = standard_scale(target_k)
    target_t_stand = standard_scale(target_t)

    seq_aa_processed = tokenize_and_pad_sequences(seq_aa, num_words=N_CHAR, max_len=SEQ_LENGTH)

    return {"sequences": seq_aa_processed,
            "trypsin_stability": target_t_stand,
            "chemotrypsin_stability": target_k_stand,
            }


