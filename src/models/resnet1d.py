from tensorflow.keras import layers
import tensorflow as tf


class ProtResNet_1D(tf.keras.Model):

    def __init__(
            self,
            num_char,
            num_conv_layers: int = 2,
            kernel_size: int = 3,
            strides: int = 1,
            target_names=None,
            **kwargs,
    ):
        super(ProtResNet_1D, self).__init__(name="ProtResNet_1D")
        self.one_hot_layer = layers.Lambda(tf.one_hot, arguments={'depth': num_char}, output_shape=(None, 19))

    def call(self, inputs, training=None, mask=None):
        return self.one_hot_layer(inputs)



class ResidualBlock_1D(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size):
        super(ResidualBlock_1D, self).__init__()

        self.relu = layers.ReLU()
        self.batch_norm = layers.BatchNormalization()
        self.conv1d = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', strides=1)

    def __call__(self, X):
        x = self.conv1d(X)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = layers.add([x, X])

        return x
