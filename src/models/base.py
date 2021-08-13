from tensorflow.keras import layers, Model


def build_base_model(
        seq_length,
        num_char,
        num_conv_layers: int = 2,
        num_filters: int = 256,
        kernel_size: int = 3,
        strides: int = 1,
        embdedding_output_dim: int = 64,
        padding="valid",
        target_names=None,
):
    if target_names is None:
        target_names = ("Trypsin", "Chemotrypsin")

    input_layer = layers.Input(shape=(seq_length,))
    x = layers.Embedding(num_char, embdedding_output_dim)(input_layer)

    for _ in range(num_conv_layers):
        x = layers.Conv1D(num_filters, kernel_size, strides=strides, activation='relu', padding=padding)(x)
        x = layers.MaxPooling1D(3)(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)

    output_1 = layers.Dense(1, name=target_names[0])(x)
    output_2 = layers.Dense(1, name=target_names[1])(x)

    model = Model(inputs=input_layer, outputs=[output_1, output_2])

    return model


