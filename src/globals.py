
DATA_PATH = "../../data/210728_scrambles_for_unstructure_model.csv"
SAVE_DIR =  "../../results"

DIM = 50
N_CHAR = 19
SEQ_LENGTH = 72

PARAMS = {
    "BASE_MODEL":
        {
            "strides": 1,
            "padding": "causal",
            "num_filters": 256,
            "num_conv_layers": 1,
            "kernel_size": 12,
            "embdedding_output_dim": 64,
            "epochs": 7,
        }
}

PARAMETER_GRID = {
    "BASE_MODEL":
        {
            "num_conv_layers": [1, 2, 3, 4,],
            "padding": ["valid", "causal"],
            "num_filters": [8, 16, 32, 64, 128, 256, 512, 1028],
            "kernel_size": [3, 5, 8, 10, 12, 15],
            "strides": [1, 2, 3],
            "embdedding_output_dim": [4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
}
