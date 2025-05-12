import typing
import tensorflow as tf


def get_model(
    sample_shape: typing.Tuple[int], num_classes: int = 2, use_softmax: bool = True
) -> tf.keras.Model:
    """
    Trả về một model đơn giản gồm 3 Dense layers.

    Args:
        sample_shape (tuple): shape của input (không gồm batch)
        num_classes (int): số lớp output, mặc định 2
        use_softmax (bool): nếu True → activation cuối là softmax (multi-class)
                            nếu False → sigmoid (binary)

    Returns:
        tf.keras.Model: mô hình đã tạo, CHƯA COMPILE
    """
    inputs = tf.keras.Input(shape=sample_shape)
    x = tf.keras.layers.Dense(100, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)

    if use_softmax:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
