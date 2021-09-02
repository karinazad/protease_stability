from tensorflow.keras import layers
import tensorflow as tf


class ProtConvNet1D(tf.keras.Model):

    def __init__(
            self,
            num_char,
            num_conv_layers: int = 2,
            num_dense_layers: int = 2,
            num_units_dense_layer: int = 256,
            kernel_size: int = 3,
            strides: int = 1,
            embdedding_output_dim: int = 64,
            target_names=None,
            **kwargs,
    ):
        super(ProtConvNet1D, self).__init__(name="ProtConvNet1D")

        if target_names is None:
            target_names = ("Target 1", "Target 2")

        filter_sizes = [256, 128, 64, 32, 16]
        assert num_conv_layers < len(filter_sizes) - 1

        self.embedding_layer = layers.Embedding(num_char, embdedding_output_dim)
        self.conv_layers = [
            layers.Conv1D(filter_sizes[i], kernel_size, strides=strides, activation='relu')
            for i in range(num_conv_layers)
        ]
        self.maxpool_layer = layers.MaxPooling1D(3)
        self.global_maxpool_layer = layers.GlobalMaxPooling1D()

        self.dense_layers = [
            layers.Dense(num_units_dense_layer, activation='relu')
            for _ in range(num_dense_layers)
        ]
        self.output_layer1 = layers.Dense(1, name=target_names[0])
        self.output_layer2 = layers.Dense(1, name=target_names[1])

    def call(self, inputs, training=None, mask=None):

        x = self.embedding_layer(inputs)

        for layer in self.conv_layers:
            x = layer(x)
            x = self.maxpool_layer(x)

        x = self.global_maxpool_layer(x)

        for layer in self.dense_layers:
            x = layer(x)

        output1 = self.output_layer1(x)
        output2 = self.output_layer2(x)

        return [output1, output2]



