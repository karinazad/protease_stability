from tensorflow.keras import layers
import tensorflow as tf


class ProtConvNet2D(tf.keras.Model):

    def __init__(
            self,
            num_char,
            seq_length,
            num_conv_layers: int = 2,
            num_dense_layers: int = 2,
            num_units_dense_layer: int = 256,
            kernel_size: int = 3,
            strides: int = 1,
            target_names=None,
            **kwargs,
    ):
        super(ProtConvNet2D, self).__init__(name="ProtConvNet2D")

        if target_names is None:
            target_names = ("Target_1", "Target_2")

        filter_sizes = [256, 128, 64, 32, 16]
        assert num_conv_layers < len(filter_sizes) - 1

        self.one_hot_layer = layers.Lambda(tf.one_hot,
                                           arguments={'depth': num_char},
                                           output_shape=num_char)
        self.expand_dim_layer = layers.Lambda(tf.expand_dims,
                                              arguments={"axis": 3})
        self.conv_layers = [
            layers.Conv2D(filters=filter_sizes[i],
                          kernel_size=(kernel_size, kernel_size),
                          input_shape=(seq_length, num_char, 1),
                          strides=strides,
                          activation="relu")
            for i in range(num_conv_layers)
        ]
        self.maxpool_layer = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense_layers = [
            layers.Dense(num_units_dense_layer, activation='relu')
            for _ in range(num_dense_layers)
        ]
        self.output_layer1 = layers.Dense(1, name=target_names[0])
        self.output_layer2 = layers.Dense(1, name=target_names[1])

    def call(self, inputs, training=None, mask=None):

        x = self.one_hot_layer(inputs)
        x = self.expand_dim_layer(x)

        for layer in self.conv_layers:
            x = layer(x)
            x = self.maxpool_layer(x)

        x = self.flatten(x)

        for layer in self.dense_layers:
            x = layer(x)

        output1 = self.output_layer1(x)
        output2 = self.output_layer2(x)

        return [output1, output2]