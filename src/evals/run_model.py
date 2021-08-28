from tensorflow.keras import optimizers

from src.utils.utils import _train_test_split
from src.evals.data_processing import get_and_process_data
from src.models.base import build_base_model
from src.globals import SEQ_LENGTH, N_CHAR, PARAMS

DATA_PATH = "data/210728_scrambles_for_unstructure_model.csv"


def run_model(data_path=DATA_PATH, model_type="base"):
    if model_type == "base":
        params = PARAMS["BASE_MODEL"]
        epochs = params.pop("epochs")

    else:
        params = None
        epochs = None

    data = get_and_process_data(data_path)
    X, y1, y2 = data["sequences"], data["trypsin_stability"], data["chemotrypsin_stability"]

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = _train_test_split(X, y1, y2)

    model = build_base_model(seq_length=SEQ_LENGTH, num_char=N_CHAR, **params)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    history = model.fit(X_train, [y1_train, y2_train],
                        epochs=epochs,
                        batch_size=128)

    loss, y1_loss, y2_loss, _, _ = model.evaluate(X_test, [y1_test, y2_test])

    print(f"\t\tSCORE: total loss = {loss}, trypsin loss = {y1_loss}, chemotrypsin loss = {y2_loss}\n\n")

    return model, history
