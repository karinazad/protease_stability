from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src_.utils.general import multi_target_train_test_split

np.random.seed(42)


def compute_stability_score(measured_EC50, predicted_EC50):
    stability_score = measured_EC50 - predicted_EC50
    return stability_score


def plot_stability_score_correlation(model, X, kT, kC, sample: Optional[int] = False, title=None, save_path=None):
    kT, kC = np.array(kT), np.array(kC)

    if sample:
        indices = np.random.randint(low=0, high=X.shape[0], size=(sample,))
        X, kT, kC = X[indices], kT[indices], kC[indices]

    kT_pred, kC_pred = model(X)
    kT_pred, kC_pred = np.array(kT_pred).flatten(), np.array(kC_pred).flatten()

    # Get the stability scores and correlations
    stability_T, stability_C = compute_stability_score(kT, kT_pred), compute_stability_score(kC, kC_pred)
    annots_r2 = [np.round(r2_score(kT, kC), 2), np.round(r2_score(stability_T, stability_C), 2)]

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(kT, kC, alpha=0.3)
    ax.set_xlabel("kT (Trypsin)")
    ax.set_ylabel("kC (Chemotrypsin)")
    ax.set_title("Unfolded")
    ax.annotate(f"R^2={annots_r2[0]}", xy=(15, 150), xycoords="axes points")

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(stability_T, stability_C, alpha=0.3, color="orange")
    ax.set_xlabel("Stability score - Trypsin")
    ax.set_ylabel("Stability score - Chemotrypsin")
    ax.set_title("Stability Score")
    ax.annotate(f"R^2={annots_r2[1]}", xy=(15, 150), xycoords="axes points")

    if title is None:
        title = model.name
    plt.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


