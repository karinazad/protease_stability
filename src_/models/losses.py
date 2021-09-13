from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf


def agreement_mse(kT_true, kT_pred, kC_true, kC_pred):
    mse = MeanSquaredError()
    stab_T = compute_stability_score(kT_true, kT_pred)
    stab_C = compute_stability_score(kC_true, kC_pred)

    return mse(stab_T, stab_C)


def combined_mse(kT_true, kT_pred, kC_true, kC_pred):
    mse = MeanSquaredError()

    mse_T = mse(kT_true, kT_pred)
    mse_C = mse(kC_true, kC_pred)

    return mse_T + mse_C


def compute_stability_score(measured, predicted):
    stability_score = measured - predicted

    return stability_score
