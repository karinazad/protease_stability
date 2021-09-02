import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def compute_stability_score(measured_EC50, predicted_EC50):
    stability_score = measured_EC50 - predicted_EC50

    return stability_score


def compute_pearsonr(x, y):
    r, p = pearsonr(x, y)
    return round(r, 3)


def plot_stability_score_correlation(model, X, kT, kC):
    kT, kC = np.array(kT), np.array(kC)

    kT_pred, kC_pred = model(X)
    kT_pred, kC_pred = np.array(kT_pred).flatten(), np.array(kC_pred).flatten()

    # Get the stability scores and correlations
    stability_T, stability_C = compute_stability_score(kT, kT_pred), compute_stability_score(kC, kC_pred)
    correlations = [compute_pearsonr(kT, kC), compute_pearsonr(stability_T, stability_C)]

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(kT, kC, alpha=0.3)
    ax.set_xlabel("Trypsin EC50")
    ax.set_ylabel("Chemotrypsin EC50")
    ax.set_title("Measured EC50")
    ax.annotate(f"pearson r={correlations[0]}", xy=(10, 150), xycoords="axes points")

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(stability_T, stability_C, alpha=0.3, color="orange")
    ax.set_xlabel("Trypsin EC50")
    ax.set_ylabel("Chemotrypsin EC50")
    ax.set_title("Stability Score\n Measured EC50 - USM_EC50")
    ax.annotate(f"pearson r={correlations[1]}", xy=(10, 150), xycoords="axes points")

    plt.suptitle(model.name, fontsize=14)
    fig.tight_layout()
    plt.show()
