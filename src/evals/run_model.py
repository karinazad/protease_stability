"""
Module to implement project-specific functions related to training and evaluation of models.
"""
import argparse
from tensorflow.keras import optimizers

from src.models.convnet1d import ProtConvNet1D
from src.models.convnet2d import ProtConvNet2D
from src.utils.general import multi_target_train_test_split
from src.evals.data_processing import get_and_process_data
from src.config import Config


def build_model(model_type: str = "convnet_1d", **kwargs):
    if model_type == "convnet_1d" or model_type == "resnet_1d":
        return ProtConvNet1D(**kwargs)

    elif model_type == "convnet_2d" or model_type == "resnet_2d":
        return ProtConvNet2D(**kwargs)

    else:
        raise NotImplementedError(f"Invalid selection: {model_type}.")


def get_params(model_type: str = "convnet_1d"):
    assert model_type in Config.config("params").keys()
    params = Config.config("params")[model_type].copy()

    try:
        epochs = params.pop("epochs")
    except KeyError:
        epochs = Config.config("default_epochs")

    return params, epochs


def run_model(data_path=None, model_type="convnet_1d", **kwargs):
    print(f"Model: {model_type.upper()}")

    if data_path is None:
        data_path = Config.config("data_path")

    X, y1, y2 = get_and_process_data(data_path)
    target_names = Config.config("target_names")

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = \
        multi_target_train_test_split(X, y1, y2, return_val=False)

    params, epochs = get_params(model_type=model_type)

    print(f"\tBuilding...", end=" ")
    model = build_model(model_type=model_type,
                        num_char=Config.config("n_char"),
                        seq_length=Config.config("seq_length"),
                        target_names=target_names,
                        **params)
    print(f"\tdone.")

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mse',
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

def save_model(model, save_dir):
    pass


def load_model(model, save_dir):
    pass

