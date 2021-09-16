from collections import defaultdict
import tensorflow as tf
import numpy as np

from src_.models.convnet1d import ProtConvNet1D
from src_.models.convnet2d import ProtConvNet2D
from src_.models.evaluator_model import EMConvNet2D
from src_.models.losses import unfolded_mse, agreement_mse
from src_.utils.general import create_train_validation_tf_dataset, create_train_tf_dataset, sample_arrays

np.random.seed(0)


class ProtNet:
    def __init__(self,
                 model_type="convnet_1d",
                 **kwargs):

        assert model_type in ["convnet_1d", "convnet_2d", "evaluator_model"], \
            f"Invalid selection: {model_type}."

        if model_type == "convnet_1d":
            self.model = ProtConvNet1D(**kwargs)
        elif model_type == "convnet_2d":
            self.model = ProtConvNet2D(**kwargs)
        elif model_type == "evaluator_model":
            self.model = EMConvNet2D(**kwargs)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.losses = defaultdict(list)

    def train(self,
              X_unfolded,
              kT_unfolded,
              kC_unfolded,
              X_folded,
              kT_folded,
              kC_folded,
              epochs=5,
              batch_size=64,
              alpha=0.5,
              validation=True,
              early_stopping=True,
              ):
        assert 0 <= alpha <= 1, "Alpha must be in the [0,1] range."

        min_val_loss = np.inf
        early_stopping_patience = 7
        val_loss_increase_count = 0

        if validation:
            train_dataset, val_dataset = create_train_validation_tf_dataset(X_unfolded, kT_unfolded, kC_unfolded,
                                                                            X_folded, kT_folded, kC_folded,
                                                                            train_batch_size=batch_size)
        else:
            train_dataset = create_train_tf_dataset(X_unfolded, kT_unfolded, kC_unfolded,
                                                    X_folded, kT_folded, kC_folded,
                                                    train_batch_size=batch_size)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}:")

            for step, data in enumerate(train_dataset):
                x, y1, y2, x_folded, y1_folded, y2_folded = data
                losses = \
                    self._train_step(x, y1, y2, x_folded, y1_folded, y2_folded, alpha=alpha)

                if step % 50 == 0:
                    self._log_steps(losses, step)

                    if validation:
                        for val_data in val_dataset:
                            x, y1, y2, x_folded, y1_folded, y2_folded = val_data
                            val_losses = \
                                self._val_step(x, y1, y2, x_folded, y1_folded, y2_folded, alpha=alpha)

                            self._log_steps(val_losses, step, validation=True)

                            val_loss = val_losses["loss"]
                            if early_stopping:
                                if val_loss > min_val_loss:
                                    val_loss_increase_count += 1

                                    if val_loss_increase_count == early_stopping_patience:
                                        return None
                                else:
                                    min_val_loss = val_loss
                                    val_loss_increase_count = 0

    def predict(self, X):
        return [y.numpy() for y in self.model(X)]

    def evaluate(self,
                 X_unfolded,
                 kT_unfolded,
                 kC_unfolded,
                 X_folded,
                 kT_folded,
                 kC_folded,
                 sample=1000,
                 ):

        X_unfolded, kT_unfolded, kC_unfolded, X_folded, kT_folded, kC_folded =\
                sample_arrays([X_unfolded, kT_unfolded, kC_unfolded, X_folded, kT_folded, kC_folded],
                              n_samples=sample)

        # Evaluate performance on predicting kT, kC for unfolded data
        # and compare to available targets
        kT_pred, kC_pred = self.predict(X_unfolded)
        mse_kT, mse_kC = unfolded_mse(kT_true=kT_unfolded, kT_pred=kT_pred,
                                      kC_true=kC_unfolded, kC_pred=kC_pred)

        # Evaluate performance on predicting kT, kC for folded data
        # and compare similarity of obtained stability scores (no ground truth data available
        kT_pred, kC_pred = self.predict(X_folded)
        agreement_loss_term = agreement_mse(kT_true=kT_folded, kT_pred=kT_pred,
                                            kC_true=kC_folded, kC_pred=kC_pred)

        mse_kT, mse_kC, agreement_loss_term = \
            list(map(lambda x: np.round(x.numpy(), 3),
                     [mse_kT, mse_kC, agreement_loss_term]))
        print(
            f"Test data evaluation: "
            f"\n\t MSE Trypsin={round(mse_kT,3)} "
            f"\n\t MSE Chemotrypsin={round(mse_kC,3)} "
            f"\n\t Stability Score MSE={round(agreement_loss_term,3)}")

        return mse_kT, mse_kC, agreement_loss_term

    def _train_step(self, x, y1, y2, x_folded, y1_folded, y2_folded, alpha):
        with tf.GradientTape() as tape:
            y1_pred, y2_pred = self.model(x, training=True)
            y1_folded_pred, y2_folded_pred = self.model(x_folded, training=True)

            y1_pred, y2_pred, y1_folded_pred, y2_folded_pred = \
                list(map(lambda a: tf.cast(a, tf.float64),
                         [y1_pred, y2_pred, y1_folded_pred, y2_folded_pred]))

            mse_kT, mse_kC = unfolded_mse(kT_true=y1, kT_pred=y1_pred, kC_true=y2, kC_pred=y2_pred)
            mse_loss_term = mse_kT + mse_kC
            agreement_loss_term = agreement_mse(kT_true=y1_folded, kT_pred=y1_folded_pred,
                                                kC_true=y2_folded, kC_pred=y2_folded_pred)

            loss = (1 - alpha) * mse_loss_term + alpha * agreement_loss_term

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {
            "loss": loss,
            "unfolded_mse": mse_loss_term,
            "agreement_loss_term": agreement_loss_term,
            "mse_kT": mse_kT,
            "mse_kC": mse_kC,
        }

    def _val_step(self, x, y1, y2, x_folded, y1_folded, y2_folded, alpha):
        y1_pred, y2_pred = self.model(x, training=True)
        y1_folded_pred, y2_folded_pred = self.model(x_folded, training=True)

        y1_pred, y2_pred, y1_folded_pred, y2_folded_pred = \
            list(map(lambda a: tf.cast(a, tf.float64),
                     [y1_pred, y2_pred, y1_folded_pred, y2_folded_pred]))

        mse_kT, mse_kC = unfolded_mse(kT_true=y1, kT_pred=y1_pred, kC_true=y2, kC_pred=y2_pred)
        mse_loss_term = mse_kT + mse_kC

        agreement_loss_term = agreement_mse(kT_true=y1_folded, kT_pred=y1_folded_pred,
                                            kC_true=y2_folded, kC_pred=y2_folded_pred)

        loss = (1 - alpha) * mse_loss_term + alpha * agreement_loss_term

        return {
            "loss": loss,
            "mse_loss_term": mse_loss_term,
            "agreement_loss_term": agreement_loss_term,
            "mse_kT": mse_kT,
            "mse_kC": mse_kC,
        }

    def _log_steps(self, losses, step, validation=False):

        if validation:
            losses = {f"val_{loss_name}": np.round(loss_value.numpy(), 3)
                      for loss_name, loss_value in losses.items()}
        else:
            losses = {f"{loss_name}": np.round(loss_value.numpy(), 3)
                      for loss_name, loss_value in losses.items()}

        print("-", end=" ")

        for loss_name, loss_value in losses.items():
            self.losses[loss_name].append(loss_value)

