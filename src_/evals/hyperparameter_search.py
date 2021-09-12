import os
import json
import sys
import pprint
sys.path.append(os.path.dirname(os.getcwd()))

import argparse
from datetime import date
import pprint

from tensorflow.keras import optimizers
import tensorflow as tf

from src_.utils.general import multi_target_train_test_split, sample_hyperparameters
from src_.evals.data_processing import get_and_process_data
from src_.evals.run_model import build_model
from src_.config import Config


def run_hyperparameter_search(parameter_grid,
                              n_iter,
                              data_path,
                              save_dir,
                              model_type="convnet_1d",
                              suffix=""):

    pp = pprint.PrettyPrinter(indent=4)
    target_names = Config.get("target_names")

    X, y1, y2 = get_and_process_data(data_path)
    X_train, _, X_val, y1_train, _, y1_val, y2_train, _, y2_val = multi_target_train_test_split(X=X,
                                                                                                y1=y1,
                                                                                                y2=y2,
                                                                                                return_val=True)
    sampled_parameters = sample_hyperparameters(parameters=parameter_grid, n_iter=n_iter)
    scores = {}

    print("\n\n\t\t\t------ HYPERPARAMETER SEARCH ------")
    for i in range(n_iter):
        print(f"\n\nTRIAL {i + 1}/{n_iter}: ")

        try:
            params = sampled_parameters[i]
            print(params)
            model = build_model(model_type=model_type,
                                seq_length=Config.get("seq_length"),
                                num_char=Config.get("n_char"),
                                target_names=Config.get("target_names"),
                                **params)

            model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mse'])

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=3)

            history = model.fit(X_train, [y1_train, y2_train],
                                validation_data=(X_val, [y1_val, y2_val]),
                                epochs=Config.get("default_epochs"),
                                batch_size=128,
                                callbacks=[callback],
                                verbose=0)

            loss, y1_loss, y2_loss, _, _ = model.evaluate(X_val,
                                                          [y1_val, y2_val])

            results = {"loss": round(loss, 3),
                       f"loss {target_names[0]}": round(y1_loss, 3),
                       f"loss {target_names[1]}": round(y2_loss, 3),
                       "hyperparameters": params,
                       "epochs": len(history.history["loss"])}

            pp.pprint(results)
            scores[i] = results
            sorted_scores = dict(list(sorted(scores.items(), key=lambda run: -run[1]["loss"], reverse=True, )))

            save_dir_model = f"{save_dir}/{model_type}"

            if not os.path.exists(save_dir_model):
                os.makedirs(save_dir_model)

            with open(f"{save_dir_model}/hyperparameters{suffix}.json", "w") as result_file:
                result_file.write(json.dumps(sorted_scores, indent=4, default=str))

        except (ValueError, RuntimeError, AssertionError) as e:
            print("\t\tAborting current run due to an error (invalid combination of parameters or encountered NaNs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter-grid",
        type=str,
        default=Config.get("parameter_grid"),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="convnet_1d",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=Config.get("data_path"),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=Config.get("save_dir"),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=f"-{date.today()}",
    )
    args = parser.parse_args()

    run_hyperparameter_search(parameter_grid=args.parameter_grid[args.model_type],
                              n_iter=args.num_iters,
                              data_path=args.data_path,
                              save_dir=os.path.join(args.save_dir, "hyperparameters"),
                              model_type=args.model_type,
                              suffix=args.suffix)
