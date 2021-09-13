"""
Module to implement project-specific functions related to training and evaluation of models.
"""
import os

from tensorflow.keras import optimizers

from src_.models.convnet1d import ProtConvNet1D
from src_.models.convnet2d import ProtConvNet2D
from src_.models.evaluator_model import EMConvNet2D
from src_.utils.general import multi_target_train_test_split
from src_.evals.data_processing import get_and_process_data
from src_.config import Config


def build_model(model_type: str = "convnet_1d", **kwargs):
    if model_type == "convnet_1d" or model_type == "resnet_1d":
        return ProtConvNet1D(**kwargs)

    elif model_type == "convnet_2d" or model_type == "resnet_2d":
        return ProtConvNet2D(**kwargs)

    elif model_type == "evaluator_model":
        return EMConvNet2D(**kwargs)

    else:
        raise NotImplementedError(f"Invalid selection: {model_type}.")


def get_params(model_type: str = "convnet_1d"):
    assert model_type in Config.get("params").keys()
    params = Config.get("params")[model_type].copy()

    try:
        epochs = params.pop("epochs")
    except KeyError:
        epochs = Config.get("default_epochs")

    return params, epochs


def run_model(
        data_path=None,
        model_type="convnet_1d",
        loss="mse",
        **kwargs,
):

    print(f"Model: {model_type.upper()}")

    if data_path is None:
        data_path = Config.get("data_path")

    X, y1, y2 = get_and_process_data(data_path)
    target_names = Config.get("target_names")

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = \
        multi_target_train_test_split(X, y1, y2, return_val=False)

    params, epochs = get_params(model_type=model_type)

    print(f"\tBuilding...", end=" ")
    model = build_model(model_type=model_type,
                        num_char=Config.get("n_char"),
                        seq_length=Config.get("seq_length"),
                        target_names=target_names,
                        **params)
    print(f"\tdone.")

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=loss,
                  metrics=['mse'])

    print("\tRunning...", end=" ")
    history = model.fit(X_train, [y1_train, y2_train],
                        epochs=epochs,
                        batch_size=128,
                        **kwargs)
    print(f"\tdone.")

    print("\tEvaluating...")
    loss, y1_loss, y2_loss, _, _ = model.evaluate(X_test, [y1_test, y2_test])

    print(f"\t\tSCORE: total loss = {loss},"
          f" {target_names[0]} loss = {y1_loss},"
          f" {target_names[1]} loss = {y2_loss}\n\n")

    return model, history


def load_saved_model(model_type, save_dir="results/saved_weights/", data_path=None):
    if data_path is None:
        data_path = Config.get("data_path")

    X, y1, y2 = get_and_process_data(data_path)
    params, epochs = get_params(model_type=model_type)

    model = build_model(
        model_type=model_type,
        num_char=Config.get("n_char"),
        seq_length=Config.get("seq_length"),
        target_names=Config.get("target_names"),
        **params
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mse']
    )

    # Call model once on data to initialize weights
    # (according to Keras, "Call the Model first, then load the weights")
    _ = model(X[:5])

    path_to_saved_weights = os.path.join(save_dir, f"{model_type}.h5")
    model.load_weights(path_to_saved_weights)

    return model
