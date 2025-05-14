import tensorflow as tf
from tensorflow.keras import backend as K

def fl_iids_loss(y_true, y_pred):
    """
    Custom FL-IIDS loss: tăng trọng số cho mẫu tấn công (label=1).
    """
    # Cast về float
    y_true = tf.cast(y_true, tf.float32)
    # Tính binary crossentropy
    bce = K.binary_crossentropy(y_true, y_pred)
    # Gán weight: 2.0 cho attack, 1.0 cho normal
    weights = tf.where(tf.equal(y_true, 1), 2.0, 1.0)
    # Trả về loss trung bình có trọng số
    return K.mean(bce * weights)
    