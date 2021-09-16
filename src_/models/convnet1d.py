import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from src_.utils.general import compute_layer_sizes


class ProtConvNet1D(tf.keras.Model):

    def __init__(
            self,
            num_char,
            num_conv_layers: int = 2,
            num_dense_layers: int = 2,
            num_units_dense_layer: int = 256,
            max_num_filters: int = 256,
            kernel_size: int = 3,
            strides: int = 1,
            embdedding_output_dim: int = 12,
            target_names=None,
            **kwargs,
    ):
        super(ProtConvNet1D, self).__init__(name="ProtConvNet1D")

        if target_names is None:
            target_names = ("target_1", "target_2")

        filter_sizes = compute_layer_sizes(max_num_filters, num_conv_layers)

        self.embedding_layer = layers.Embedding(num_char, embdedding_output_dim)
        self.conv_layers = [
            layers.Conv1D(filter_sizes[i], kernel_size, strides=strides, activation='relu', padding="same")
            for i in range(num_conv_layers)
        ]

        self.maxpool_layer = layers.MaxPooling1D(3, padding="same")
        self.global_maxpool_layer = layers.GlobalMaxPooling1D()

        self.dense_layers = [
            layers.Dense(num_units_dense_layer, activation='relu')
            for _ in range(num_dense_layers-1)
        ]

        self.dense1 = layers.Dense(num_units_dense_layer, name=target_names[0],
                                   activity_regularizer=regularizers.l2(1e-5))
        self.dense2 = layers.Dense(num_units_dense_layer, name=target_names[0],
                                   activity_regularizer=regularizers.l2(1e-5))

        self.output_layer1 = layers.Dense(1, name=target_names[0], activity_regularizer=regularizers.l2(1e-5))
        self.output_layer2 = layers.Dense(1, name=target_names[1], activity_regularizer=regularizers.l2(1e-5))

    def call(self, inputs, training=None, mask=None):
        # print("Input", inputs.shape)

        x = self.embedding_layer(inputs)
        # print("Embedding", x.shape)

        for layer in self.conv_layers:
            x = layer(x)
            # print("Conv Layer", x.shape)
            x = self.maxpool_layer(x)
            # print("Max Pool Layer", x.shape)

        x = self.global_maxpool_layer(x)
        # print("Global pool", x.shape)

        for layer in self.dense_layers:
            x = layer(x)
            # print("Dense", x.shape)

        x1 = self.dense1(x)
        x2 = self.dense2(x)

        output1 = self.output_layer1(x1)
        output2 = self.output_layer2(x2)
        # print("Output", output1.shape, "\n")

        return [output1, output2]
