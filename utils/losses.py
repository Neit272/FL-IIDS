import tensorflow as tf

# Hệ số trọng số loss
GAMMA_LGB = 0.1
GAMMA_LLS = 0.1


@tf.function
def calculate_lls_loss(y_true_one_hot, y_pred_current, y_pred_old, old_class_indices):
    """
    Sample Label Smoothing Loss (Knowledge Distillation).
    Lấy soft label từ model cũ (theta_best), ép model mới học lại class cũ.
    """
    if (
        old_class_indices is None
        or tf.size(old_class_indices) == 0
        or y_pred_old is None
    ):
        return tf.constant(0.0, dtype=tf.float32)

    y_pred_old = tf.cast(y_pred_old, dtype=y_true_one_hot.dtype)
    y_pred_current = tf.cast(y_pred_current, dtype=y_true_one_hot.dtype)
    num_classes = tf.shape(y_true_one_hot)[1]

    # Step 1: lấy output old model với các class cũ
    old_preds = tf.gather(y_pred_old, old_class_indices, axis=1)

    # Step 2: tạo mask cho class cũ / mới
    old_mask = tf.reduce_sum(
        tf.one_hot(old_class_indices, depth=num_classes, dtype=y_true_one_hot.dtype),
        axis=0,
    )
    new_mask = 1.0 - old_mask

    # Step 3: giữ y_true cho class mới
    y_new = y_true_one_hot * new_mask

    # Step 4: tạo nhãn mềm từ old_preds cho class cũ
    batch_size = tf.shape(y_true_one_hot)[0]
    row_idx = tf.repeat(tf.range(batch_size), tf.size(old_class_indices))
    col_idx = tf.tile(old_class_indices, [batch_size])
    indices = tf.stack([row_idx, col_idx], axis=1)
    updates = tf.reshape(old_preds, [-1])
    y_old_soft = tf.scatter_nd(indices, updates, tf.shape(y_true_one_hot))

    # Step 5: gộp lại thành smoothed_label
    smoothed_label = y_new + y_old_soft

    # Step 6: tính KL divergence loss
    kl = tf.keras.losses.KLDivergence()
    loss = kl(smoothed_label, y_pred_current)
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

    return tf.reduce_mean(loss)


@tf.function
def calculate_lgb_proxy_loss(y_true_one_hot, y_pred_current, sample_info):
    """
    Class Gradient Balance Proxy Loss.
    Gán trọng số cao hơn cho mẫu từ memory (class cũ), thấp hơn cho mẫu mới.
    """
    is_old_mask = sample_info.get(
        "is_old", tf.zeros(tf.shape(y_true_one_hot)[0], dtype=tf.bool)
    )

    weight_new = 1.0
    weight_old = 1.5

    sample_weights = tf.where(is_old_mask, weight_old, weight_new)
    sample_weights = tf.cast(sample_weights, dtype=y_pred_current.dtype)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none")
    loss = cce(y_true_one_hot, y_pred_current)
    weighted_loss = loss * sample_weights

    return tf.reduce_mean(weighted_loss)
