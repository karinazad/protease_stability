"""
Global variables to be used in the project. Set in Config class.
"""

PARAMS = {
    "convnet_1d":
        {
            "epochs": 13,
            "strides": 1,
            "kernel_size": 10,
            "num_units_dense_layer": 64,
            "num_dense_layers": 1,
            "embdedding_output_dim": 12,
        },
    "convnet_2d":
        {
            "epochs": 7,
            "strides": 1,
            "padding": "causal",
            "num_filters": 256,
            "num_conv_layers": 1,
            "kernel_size": 12,
            "embdedding_output_dim": 64,

        }
}

PARAMETER_GRID = {
    "convnet_1d":
        {
            "num_conv_layers": [1, 2, 3],
            "kernel_size": [3, 5, 8, 10, 12, 15],
            "strides": [1],
            "embdedding_output_dim": [4, 6, 8, 10, 12, 14, 16, 18, 20],
            "num_dense_layers": [1, 2, 3, 4, ],
            "num_units_dense_layer": [16, 32, 64, 128, 256],
        },
    "convnet_2d":
        {
            "num_conv_layers": [1, 2, 3],
            "kernel_size": [3, 5, 8, 10, 12, 15],
            "strides": [1],
            "num_dense_layers": [1, 2, 3, 4, ],
            "num_units_dense_layer": [16, 32, 64, 128, 256],
        },
}


class Config:
    __conf = {
        "data_path": "data/210728_scrambles_for_unstructure_model.csv",
        "save_dir": "/results",
        "target_names": ["Trypsin", "Chemotrypsin"],
        "dim": 50,
        "n_char": 19,
        "seq_length": 72,
        "default_epochs": 7,
        "params": PARAMS,
        "parameter_grid": PARAMETER_GRID,
    }

    __setters = ["data_path", "save_dir", "params", "parameter_grid"]

    @staticmethod
    def config(name):
        return Config.__conf[name]

    @staticmethod
    def get(name):
        return Config.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Config.__setters:
            Config.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
