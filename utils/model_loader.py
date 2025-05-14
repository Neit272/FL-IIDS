import typing
import tensorflow as tf
from utils.losses import fl_iids_loss

use_fl_iids_loss = True

def get_model(sample_shape: typing.Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(sample_shape)
    x = tf.keras.layers.Dense(100, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    loss_fn = fl_iids_loss if use_fl_iids_loss else "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=loss_fn,
        metrics=["binary_accuracy"]
    )

    print(f"Compiled model with loss: {'fl_iids_loss' if use_fl_iids_loss else 'binary_crossentropy'}")
    return model