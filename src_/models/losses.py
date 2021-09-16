from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np

from src_.config import Config


def agreement_mse(kT_true, kT_pred, kC_true, kC_pred):
    mse = MeanSquaredError()
    stab_T = compute_stability_score(kT_true, kT_pred)
    stab_C = compute_stability_score(kC_true, kC_pred)

    return mse(stab_T, stab_C)


def unfolded_mse(kT_true, kT_pred, kC_true, kC_pred):

    # if Config.get("clip_predictions_unfolded_loss", False):
    #     range_kT = Config.get("range_kT")
    #     range_kC = Config.get("range_kC")
    #
    #     kT_pred = np.clip(kT_pred, range_kT[0], range_kT[1])
    #     kC_pred = np.clip(kC_pred, range_kC[0], range_kC[1])

    mse = MeanSquaredError()
    mse_T = mse(kT_true, kT_pred)
    mse_C = mse(kC_true, kC_pred)

    return mse_T, mse_C


def compute_stability_score(measured, predicted):
    stability_score = measured - predicted

    return stability_score


