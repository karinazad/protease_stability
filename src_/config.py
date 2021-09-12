"""
Global variables to be used in the project. Set in Config class.
"""

PARAMS = {
    "convnet_1d":
        {
            "epochs": 8,
            "num_units_dense_layer": 64,
            "num_dense_layers": 5,
            "num_conv_layers": 4,
            "max_num_filters": 128,
            "kernel_size": 12,
            "embdedding_output_dim": 12
        },

    "convnet_2d":
        {
            "epochs": 13,
            "strides": 1,
            "num_units_dense_layer": 16,
            "num_dense_layers": 4,
            "num_conv_layers": 3,
            "max_num_filters": 256,
            "kernel_size": 3
        },

    "evaluator_model":
        {
            "epochs": 15,
        },
}



PARAMETER_GRID = {
    "convnet_1d":
        {
            "num_conv_layers": [1, 2, 3, 4],
            "num_dense_layers": [1, 2, 3, 4, 5],
            "num_units_dense_layer": [16, 32, 64, 128, 256],
            "max_num_filters": [32, 64, 128, 256, 512, 1028],
            "kernel_size": [2, 3, 5, 8, 10, 12, 15],
            "strides": [1],
            "embdedding_output_dim": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],

        },
    "convnet_2d":
        {
            "num_conv_layers": [1, 2, 3, 4],
            "num_dense_layers": [1, 2, 3, 4, 5],
            "num_units_dense_layer": [16, 32, 64, 128, 256],
            "max_num_filters": [32, 64, 128, 256, 512, 1028],
            "kernel_size": [2, 3, 5, 8, 10, 12, 15],
            "strides": [1],
        },
}


class Config:
    __conf = {
        "data_path": "data/210728_scrambles_for_unstructure_model.csv",
        "save_dir": "/results",
        "target_names": ["Trypsin", "Chemotrypsin"],
        "n_char": 21,
        "seq_length": 74,
        "default_epochs": 20,
        "params": PARAMS,
        "parameter_grid": PARAMETER_GRID,
    }

    __setters = ["data_path", "save_dir", "params", "parameter_grid"]

    @staticmethod
    def get(name):
        return Config.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Config.__setters:
            Config.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
