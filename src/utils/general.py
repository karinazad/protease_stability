import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ParameterSampler


def multi_target_train_test_split(X, y1, y2, random_state=42, return_val = True):
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2, test_size=0.15, random_state=random_state)

    if return_val is False:
        return X_train, X_test, y1_train, y1_test, y2_train, y2_test

    X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
        X_train, y1_train, y2_train, random_state=random_state)

    return X_train, X_test, X_val, y1_train, y1_test, y1_val, y2_train, y2_test, y2_val


def standard_scale(df):
    if df.ndim == 1:
        X = df.values.reshape(-1, 1)
        columns = [df.name]
    else:
        X = df.values
        columns = df.columns

    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)
    df_processed = pd.DataFrame(X_processed, columns=columns, index=df.index)

    return df_processed


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def tokenize_and_pad_sequences(data, num_words, max_len):
    data_S = data.apply(lambda x: " ".join(list(x)))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data_S)
    transformed = tokenizer.texts_to_sequences(data_S)

    transformed = pad_sequences(transformed, maxlen=max_len, padding='post')

    return transformed


def sample_hyperparameters(parameters, n_iter):
    return list(
        ParameterSampler(parameters, n_iter=n_iter)
    )

def drop_nans(X, targets):
    indices_to_remove = X[X.isnull()].index

    X_ = X.drop(indices_to_remove, axis=0)
    targets_ = [
        target.drop(indices_to_remove, axis=0) for target in targets
    ]

    return X_, targets_
