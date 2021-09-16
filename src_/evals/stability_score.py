from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src_.utils.general import sample_arrays

np.random.seed(42)


def compute_stability_score(measured_EC50, predicted_EC50):
    stability_score = measured_EC50 - predicted_EC50
    return stability_score


def get_predictions_and_stability_scores(model, X, kT, kC, sample=None):
    X, kT, kC = sample_arrays([X, kT, kC], n_samples=sample)
    kT_pred, kC_pred = model(X)
    kT_pred, kC_pred = np.array(kT_pred).flatten(), np.array(kC_pred).flatten()
    stability_T, stability_C = compute_stability_score(kT, kT_pred), compute_stability_score(kC, kC_pred)

    return kT_pred, kC_pred, stability_T, stability_C


def plot_stability_score_correlation(model, X, kT, kC, sample=None, title=None, save_path=None):
    X, kT, kC = sample_arrays([X, kT, kC], n_samples=sample)

    # Get stability scores and correlations
    kT_pred, kC_pred, stability_T, stability_C = get_predictions_and_stability_scores(model, X, kT, kC)
    annots_r2 = [np.round(pearsonr(kT, kC)[0], 2), np.round(pearsonr(stability_T, stability_C)[0], 2)]

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(kT, kC, alpha=0.3)
    ax.set_xlabel("kT (Trypsin)")
    ax.set_ylabel("kC (Chemotrypsin)")
    ax.set_title("Measured kT/kC")
    ax.annotate(f"r^2={annots_r2[0]}", xy=(15, 150), xycoords="axes points")

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(stability_T, stability_C, alpha=0.3, color="orange")
    ax.set_xlabel("Stability score Trypsin")
    ax.set_ylabel("Stability score Chemotrypsin")
    ax.set_title("Stability Scores")
    ax.annotate(f"r^2={annots_r2[1]}", xy=(15, 150), xycoords="axes points")

    if title is None:
        title = model.name
    plt.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
