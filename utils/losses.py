# --- START OF FILE utils/losses.py ---

import tensorflow as tf

# Hệ số cho loss function kết hợp (Eq. 9). Bạn có thể điều chỉnh các giá trị này.
# GAMMA_LGB: Trọng số cho Class Gradient Balance Loss (hoặc proxy của nó)
# GAMMA_LLS: Trọng số cho Sample Label Smoothing Loss (Knowledge Distillation)
GAMMA_LGB = 0.5
GAMMA_LLS = 0.5

# Hàm tính LLS (Eq. 8) - Knowledge Distillation Loss
@tf.function # Sử dụng tf.function để tối ưu hóa
def calculate_lls_loss(y_true_one_hot, y_pred_current, y_pred_old, old_class_indices):
    """
    Tính Sample Label Smoothing Loss (Knowledge Distillation).
    Hàm này giúp model mới học lại kiến thức về các lớp cũ từ model cũ tốt nhất.

    Args:
        y_true_one_hot: Nhãn one-hot gốc của batch dữ liệu kết hợp (mới + cũ từ memory).
                        Shape: (batch_size, num_classes)
        y_pred_current: Dự đoán (softmax output) của model hiện tại đang huấn luyện.
                        Shape: (batch_size, num_classes)
        y_pred_old:     Dự đoán (softmax output) của model cũ tốt nhất (đã được lưu).
                        Shape: (batch_size, num_classes)
        old_class_indices: Tensor 1D chứa các chỉ số (integer) của các lớp được coi là "cũ"
                           trong ngữ cảnh của task hiện tại. Ví dụ: tf.constant([0, 1, 2], dtype=tf.int32)

    Returns:
        LLS loss (scalar tensor). Trả về 0.0 nếu không có lớp cũ.
    """
    # Nếu không có lớp cũ hoặc không có dự đoán cũ, không tính LLS
    if old_class_indices is None or tf.size(old_class_indices) == 0 or y_pred_old is None:
        return tf.constant(0.0, dtype=tf.float32)

    # Đảm bảo kiểu dữ liệu nhất quán
    y_pred_old = tf.cast(y_pred_old, dtype=y_true_one_hot.dtype)
    y_pred_current = tf.cast(y_pred_current, dtype=y_true_one_hot.dtype)

    # --- Tạo "soft label" theo Fig. 2 trong bài báo FL-IIDS ---
    # 1. Lấy các dự đoán của model cũ cho các lớp cũ
    # gather sẽ lấy các cột tương ứng với old_class_indices từ y_pred_old
    old_preds_from_old_model = tf.gather(y_pred_old, old_class_indices, axis=1)

    # 2. Tạo mask cho các lớp cũ và lớp mới trong nhãn gốc (y_true_one_hot)
    num_classes = tf.shape(y_true_one_hot)[1]
    old_class_mask = tf.reduce_sum(tf.one_hot(old_class_indices, depth=num_classes, dtype=y_true_one_hot.dtype), axis=0)
    new_class_mask = 1.0 - old_class_mask

    # 3. Giữ nguyên phần nhãn gốc cho các lớp mới
    y_true_new_part = y_true_one_hot * new_class_mask

    # 4. Tạo phần nhãn "mềm" cho các lớp cũ từ dự đoán của model cũ
    # Chúng ta cần đặt các giá trị old_preds_from_old_model vào đúng vị trí cột trong tensor mới
    # Sử dụng scatter_nd để làm việc này hiệu quả
    batch_size = tf.shape(y_true_one_hot)[0]
    # Tạo indices cho scatter_nd: [[0, old_idx1], [0, old_idx2], ..., [1, old_idx1], ...]
    row_indices = tf.repeat(tf.range(batch_size), repeats=tf.size(old_class_indices))
    col_indices = tf.tile(old_class_indices, [batch_size])
    scatter_indices = tf.stack([row_indices, col_indices], axis=1)

    # Reshape updates cho phù hợp với scatter_indices
    updates = tf.reshape(old_preds_from_old_model, [-1])

    # Tạo phần nhãn mềm cho các lớp cũ
    y_soft_old_part = tf.scatter_nd(indices=scatter_indices, updates=updates, shape=tf.shape(y_true_one_hot))

    # 5. Kết hợp phần nhãn mới và phần nhãn cũ mềm -> đây là smoothed_label
    smoothed_label = y_true_new_part + y_soft_old_part

    # (Optional) Chuẩn hóa lại smoothed_label để đảm bảo tổng là 1 nếu cần thiết
    # Tuy nhiên, KLDivergence thường xử lý tốt với các phân phối chưa chuẩn hóa nhẹ
    # smoothed_label = tf.nn.softmax(smoothed_label) # Có thể không cần thiết và làm thay đổi ý nghĩa

    # --- Tính KL Divergence ---
    # KL(P || Q) đo sự khác biệt giữa phân phối P (smoothed_label) và Q (y_pred_current)
    # Sử dụng hàm loss của Keras
    kl_loss_fn = tf.keras.losses.KLDivergence()

    # Tính loss trên từng mẫu rồi lấy trung bình
    # KLDivergence mong đợi y_true và y_pred
    lls = kl_loss_fn(smoothed_label, y_pred_current)

    # Tránh giá trị NaN hoặc Inf nếu dự đoán bằng 0 ở đâu đó P > 0
    lls = tf.where(tf.math.is_finite(lls), lls, tf.zeros_like(lls))

    return tf.reduce_mean(lls) # Trả về loss trung bình của batch

# Hàm tính LGB (Proxy) - Class Gradient Balance Loss (Phiên bản đơn giản hóa)
# Phiên bản này sử dụng trọng số cố định cho lớp cũ/mới thay vì tính toán gradient phức tạp
@tf.function # Sử dụng tf.function để tối ưu hóa
def calculate_lgb_proxy_loss(y_true_one_hot, y_pred_current, sample_info):
    """
    Tính proxy cho Class Gradient Balance Loss bằng cách áp dụng trọng số khác nhau
    cho các mẫu thuộc lớp cũ và lớp mới trong batch. Mục đích là làm chậm
    việc học lớp mới và tăng cường ghi nhớ lớp cũ.

    Args:
        y_true_one_hot: Nhãn one-hot gốc của batch dữ liệu kết hợp.
                        Shape: (batch_size, num_classes)
        y_pred_current: Dự đoán (softmax output) của model hiện tại.
                        Shape: (batch_size, num_classes)
        sample_info:    Dictionary chứa thông tin bổ sung về batch. Cần có key 'is_old'
                        là một Tensor boolean cùng kích thước batch_size, True nếu mẫu
                        đến từ DEM (lớp cũ), False nếu là mẫu mới của task hiện tại.

    Returns:
        Weighted Cross-Entropy loss (scalar tensor).
    """
    # Lấy cờ xác định mẫu cũ/mới từ sample_info
    # Mặc định là False (mẫu mới) nếu không có thông tin
    is_old_mask = sample_info.get('is_old', tf.zeros(tf.shape(y_true_one_hot)[0], dtype=tf.bool))

    # Định nghĩa trọng số (có thể điều chỉnh)
    # Ý tưởng: giảm nhẹ trọng số lớp mới, tăng trọng số lớp cũ
    weight_new = 1.0
    weight_old = 1.5 # Tăng cường học các lớp cũ từ DEM

    # Tạo tensor trọng số cho từng mẫu trong batch
    sample_weights = tf.where(is_old_mask, weight_old, weight_new)
    sample_weights = tf.cast(sample_weights, dtype=y_pred_current.dtype) # Đảm bảo cùng kiểu dữ liệu

    # Tính Categorical Cross-Entropy với trọng số mẫu
    # from_logits=False vì y_pred_current giả định là output của softmax
    cce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE) # Không tự động reduce

    # Tính loss trên từng mẫu
    per_sample_loss = cce_loss_fn(y_true_one_hot, y_pred_current)

    # Áp dụng trọng số
    weighted_per_sample_loss = per_sample_loss * sample_weights

    # Lấy trung bình loss của batch
    final_loss = tf.reduce_mean(weighted_per_sample_loss)

    return final_loss

# --- END OF FILE utils/losses.py ---