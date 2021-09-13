from collections import defaultdict
import tensorflow as tf
import numpy as np

from src_.models.convnet1d import ProtConvNet1D
from src_.models.convnet2d import ProtConvNet2D
from src_.models.evaluator_model import EMConvNet2D
from src_.models.losses import combined_mse, agreement_mse
from src_.utils.general import create_train_validation_tf_dataset, create_train_tf_dataset

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

        if validation:
            train_dataset, val_dataset = create_train_validation_tf_dataset(X_unfolded, kT_unfolded, kC_unfolded,
                                                                            X_folded, kT_folded, kC_folded,
                                                                            train_batch_size=batch_size)
        else:
            train_dataset = create_train_tf_dataset(X_unfolded, kT_unfolded, kC_unfolded,
                                                    X_folded, kT_folded, kC_folded,
                                                    train_batch_size=batch_size)

        if early_stopping:
            min_val_loss = np.inf
            early_stopping_patience = 5
            val_loss_increase_count = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}:")

            for step, data in enumerate(train_dataset):
                x, y1, y2, x_folded, y1_folded, y2_folded = data
                loss, mse_loss_term, agreement_loss_term = \
                    self.train_step(x, y1, y2, x_folded, y1_folded, y2_folded, alpha=alpha)

                if step % 50 == 0:
                    print(f"\n\tstep={step}:   "
                          f"loss={round(float(loss), 3)},   "
                          f"unfolded_mse={round(float(mse_loss_term), 3)},   "
                          f"stab_score_agreement_mse={round(float(agreement_loss_term), 3)}")

                    self.losses["loss"].append(loss.numpy())
                    self.losses["unfolded_mse"].append(mse_loss_term.numpy())
                    self.losses["agreement_mse"].append(agreement_loss_term.numpy())

                    if validation:
                        for val_data in val_dataset:
                            x, y1, y2, x_folded, y1_folded, y2_folded = val_data
                            loss, mse_loss_term, agreement_loss_term = \
                                self.val_step(x, y1, y2, x_folded, y1_folded, y2_folded, alpha=alpha)

                            print(f"\t"
                                  f"val_loss={round(float(loss), 3)},   "
                                  f"val_unfolded_mse={round(float(mse_loss_term), 3)},   "
                                  f"val_stab_score_agreement_mse={round(float(agreement_loss_term), 3)}")

                            self.losses["val_loss"].append(loss.numpy())
                            self.losses["val_unfolded_mse"].append(mse_loss_term.numpy())
                            self.losses["val_agreement_mse"].append(agreement_loss_term.numpy())

                            if early_stopping:
                                if loss > min_val_loss:
                                    val_loss_increase_count += 1

                                    if val_loss_increase_count == early_stopping_patience:
                                        return None
                                else:
                                    min_val_loss = loss
                                    val_loss_increase_count = 0

    def predict(self, X):
        return [y.numpy() for y in self.model(X)]

    def train_step(self, x, y1, y2, x_folded, y1_folded, y2_folded, alpha):
        with tf.GradientTape() as tape:
            y1_pred, y2_pred = self.model(x, training=True)
            y1_folded_pred, y2_folded_pred = self.model(x_folded, training=True)

            y1_pred, y2_pred, y1_folded_pred, y2_folded_pred = \
                list(map(lambda a: tf.cast(a, tf.float64),
                         [y1_pred, y2_pred, y1_folded_pred, y2_folded_pred]))

            mse_loss_term = combined_mse(kT_true=y1, kT_pred=y1_pred, kC_true=y2, kC_pred=y2_pred)
            agreement_loss_term = agreement_mse(kT_true=y1_folded, kT_pred=y1_folded_pred,
                                                kC_true=y2_folded, kC_pred=y2_folded_pred)

            loss = (1 - alpha) * mse_loss_term + alpha * agreement_loss_term

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, mse_loss_term, agreement_loss_term

    def val_step(self, x, y1, y2, x_folded, y1_folded, y2_folded, alpha):
        y1_pred, y2_pred = self.model(x, training=True)
        y1_folded_pred, y2_folded_pred = self.model(x_folded, training=True)

        y1_pred, y2_pred, y1_folded_pred, y2_folded_pred = \
            list(map(lambda a: tf.cast(a, tf.float64),
                     [y1_pred, y2_pred, y1_folded_pred, y2_folded_pred]))

        mse_loss_term = combined_mse(kT_true=y1, kT_pred=y1_pred, kC_true=y2, kC_pred=y2_pred)
        agreement_loss_term = agreement_mse(kT_true=y1_folded, kT_pred=y1_folded_pred,
                                            kC_true=y2_folded, kC_pred=y2_folded_pred)

        loss = (1 - alpha) * mse_loss_term + alpha * agreement_loss_term

        return loss, mse_loss_term, agreement_loss_term


