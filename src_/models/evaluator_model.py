from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf




class EMConvNet2D(tf.keras.Model):

    def __init__(
            self,
            num_char,
            seq_length,
            target_names=None,
            **kwargs,
    ):
        super(EMConvNet2D, self).__init__(name="EMConvNet2D")

        if target_names is None:
            target_names = ("Target_1", "Target_2")

        self.one_hot_layer = layers.Lambda(tf.one_hot, arguments={'depth': num_char}, output_shape=num_char)
        self.expand_dim_layer = layers.Lambda(tf.expand_dims, arguments={"axis": 3})

        filter_sizes = [400, 200, 100]
        kernel_sizes = [(23, 5), (1, 9), (1, 17)]
        dense_sizes = [80, 40]

        self.conv_layers = [
            layers.Conv2D(filters=filter_sizes[i],
                          kernel_size=kernel_sizes[i],
                          input_shape=(seq_length, num_char, 1),
                          activation="relu",
                          padding="same")
            for i in range(len(filter_sizes))
        ]
        self.maxpool_layer = layers.MaxPooling2D((2, 2), padding="same")
        self.dropout = layers.Dropout(rate=0.2)
        self.flatten = layers.Flatten()
        self.dense_layers = [
            layers.Dense(units, activation='relu')
            for units in dense_sizes
        ]
        self.output_layer1 = layers.Dense(1, name=target_names[0], activity_regularizer=regularizers.l2(1e-5))
        self.output_layer2 = layers.Dense(1, name=target_names[1], activity_regularizer=regularizers.l2(1e-5))

    def call(self, inputs, training=None, mask=None):
        # print("Input", inputs.shape)

        x = self.one_hot_layer(inputs)
        # print("One hot", x.shape)

        x = self.expand_dim_layer(x)
        # print("Expand dim", x.shape)

        for layer in self.conv_layers:
            x = layer(x)
            # print("Conv Layer", x.shape)
            x = self.dropout(x)
            x = self.maxpool_layer(x)
            # print("Maxpool Layer", x.shape)

        x = self.flatten(x)
        # print("Flatten", x.shape)

        for layer in self.dense_layers:
            x = layer(x)
            # print("Dense", x.shape)

        output1 = self.output_layer1(x)
        output2 = self.output_layer2(x)

        return [output1, output2]
