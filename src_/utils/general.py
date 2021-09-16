import math

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ParameterSampler


def multi_target_train_test_split(X, y1, y2, random_state=42, return_val=True):
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


def add_start_end_characters(sequence, start_char="J", end_char="O"):
    return ''.join((start_char, sequence, end_char))


def compute_layer_sizes(max_units, num_layers):
    nth_power = int(math.log2(max_units))

    if nth_power >= num_layers:
        pass
    # assert nth_power >= num_layers, \
    #     "\tToo many layers. Please provide larger size of the first layer or decrease the number of layers."

    layer_sizes = [2 ** (nth_power - i) for i in range(num_layers)]

    return layer_sizes


def create_train_validation_tf_dataset(XA, y1A, y2A, XB, y1B, y2B, val_size=0.15, train_batch_size=64):
    n_val = int(val_size * XA.shape[0])

    XA_val, y1A_val, y2A_val, XB_val, y1B_val, y2B_val = \
        list(map(lambda a: a[:n_val],
                 [XA, y1A, y2A, XB, y1B, y2B]))

    XA, y1A, y2A, XB, y1B, y2B = \
        list(map(lambda a: a[n_val:],
                 [XA, y1A, y2A, XB, y1B, y2B]))

    val_dataset = tf.data.Dataset.from_tensor_slices((XA_val, y1A_val, y2A_val, XB_val, y1B_val, y2B_val))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(XA.shape[0])

    train_dataset = tf.data.Dataset.from_tensor_slices((XA, y1A, y2A, XB, y1B, y2B))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch_size)

    return train_dataset, val_dataset


def create_train_tf_dataset(XA, y1A, y2A, XB, y1B, y2B, train_batch_size=64):
    train_dataset = tf.data.Dataset.from_tensor_slices((XA, y1A, y2A, XB, y1B, y2B))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch_size)

    return train_dataset


def average_losses(losses, epochs):
    averaged_losses = []

    for loss in losses:
        n_steps_per_epoch = int(len(loss) / epochs)
        avg_loss = [np.array(loss[i:i + 1]).mean() for i in range(n_steps_per_epoch)]

        averaged_losses.append(avg_loss)

    return averaged_losses


def sample_arrays(arrays, n_samples=None):
    np.random.seed(42)
    arrays = [np.array(arr) for arr in arrays]
    if n_samples:
        indices = np.random.randint(low=0, high=arrays[0].shape[0], size=(n_samples,))

        return list(map(lambda x: x[indices], arrays))

    return arrays


def convert_to_numpy(dfs):
    if type(dfs) is not list:
        dfs = [dfs]

    arrays = list(
        map(lambda x: np.array(x),
            dfs)
    )

    return arrays

