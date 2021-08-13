import os
import json

from tensorflow.keras import optimizers
import tensorflow as tf
from sklearn.model_selection import ParameterSampler

from src.utils.utils import _train_test_val_split
from src.data_processing import get_and_process_data
from src.models.base import build_base_model
from src.globals import SEQ_LENGTH, N_CHAR

parameter_grid = {
    "num_conv_layers": [1, 2, 3],
    "padding": ["valid", "causal"],
    "num_filters": [32, 64, 128, 256, 512, 1028],
    "kernel_size": [3, 5, 8, 10, 12, 15],
    "strides": [1, 2, 3],
    "embdedding_output_dim": [32, 64, 128, 256, 512],
}


def sample_hyperparameters(parameters, n_iter):
    return list(
        ParameterSampler(parameters, n_iter=n_iter)
    )


def run_hyperparameter_search(parameters, n_iter, data_path, save_dir, suffix=""):
    data = get_and_process_data(data_path)
    X, y1, y2 = data["sequences"], data["trypsin_stability"], data["chemotrypsin_stability"]

    (X_train, y1_train, y2_train), (X_val, y1_val, y2_val), (_, _, _) = _train_test_val_split(X, y1, y2)

    sampled_parameters = sample_hyperparameters(parameters=parameters, n_iter=n_iter)
    scores = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("------ HYPERPARAMETER SEARCH ------")
    for i in range(n_iter):
        print(f"TRIAL {i+1}/{n_iter}: ")

        try:
            params = sampled_parameters[i]
            model = build_base_model(seq_length=SEQ_LENGTH, num_char=N_CHAR, **params)
            model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            history = model.fit(X_train, [y1_train, y2_train],
                                validation_data=(X_val, [y1_val, y2_val]),
                                epochs=20,
                                batch_size=128,
                                callbacks=[callback])

            loss, y1_loss, y2_loss, _, _ = model.evaluate(X_val, [y1_val, y2_val])

            print(f"\t\tSCORE: total loss = {loss}, trypsin loss = {y1_loss}, chemotrypsin loss = {y2_loss}\n\n")

            scores[i] = {"score": loss,
                         "score trypsin": y1_loss,
                         "score chemotrypsin": y2_loss,
                         "hyperparameters": params,
                         "epochs": len(history.history["loss"])}

            sorted_scores = dict(
                list(sorted(scores.items(), key=lambda run: -run[1]["score"], reverse=True,))
            )

            with open(f"{save_dir}/hyperparameters{suffix}.json", "w") as result_file:
                result_file.write(json.dumps(sorted_scores, indent=4, default=str))

        except (ValueError, RuntimeError) as e:
            print("\t\tAborting current run due to an error:", e)







