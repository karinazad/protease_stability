import matplotlib.pyplot as plt
from src_.utils.general import sample_arrays


def plot_losses(model, selection=None):
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    if selection:
        ax.plot(model.losses[selection], label=selection)

    else:
        ax.plot(model.losses["mse_loss_term"], label="Unfolded MSE")
        ax.plot(model.losses["agreement_loss_term"], label="Stability score agreement")
        ax.plot(model.losses["loss"], label="Total loss", alpha=0.7)
    ax.set_xlabel("Steps")
    ax.set_ylabel("loss")
    ax.set_title("Train")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    if selection:
        ax.plot(model.losses[f"val_{selection}"], label=selection)

    else:
        ax.plot(model.losses["val_mse_loss_term"], label="Unfolded MSE")
        ax.plot(model.losses["val_agreement_loss_term"], label="Stability score agreement")
        ax.plot(model.losses["val_loss"], label="Total loss", alpha=0.7)
    ax.set_xlabel("Steps")
    ax.set_ylabel("loss")
    ax.set_title("Validation")
    ax.legend()

    plt.suptitle(model.model.name.replace("_", " "), fontsize=14)
    plt.show()


def plot_losses_unfolded_kT_kC(model):
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(model.losses["mse_kT"], label="Trypsin MSE")
    ax.plot(model.losses["mse_kC"], label="Chemotrypsin MSE")
    ax.set_xlabel("Steps (x50)")
    ax.set_ylabel("loss")
    ax.set_title("Train")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(model.losses["val_mse_kT"], label="Trypsin MSE")
    ax.plot(model.losses["val_mse_kC"], label="Chemotrypsin MSE")
    ax.set_xlabel("Steps")
    ax.set_ylabel("loss")
    ax.set_title("Validation")
    ax.legend()

    model_plot_name = model.model.name.replace("_", " ")
    plt.suptitle(f"{model_plot_name}: Unfolded MSE Loss", fontsize=14)
    plt.show()


def plot_scatter_predictions(model, X, kT, kC, sample=None):
    X, kT, kC = sample_arrays([X, kT, kC], n_samples=sample)
    kT_pred, kC_pred = model.predict(X)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(kT, kT_pred, alpha=0.3)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Trypsin")

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(kC, kC_pred, alpha=0.3)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Chemotrypsin")

    model_plot_name = model.model.name.replace("_", " ")
    plt.suptitle(f"{model_plot_name}", fontsize=14)
    plt.show()
