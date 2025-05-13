import pandas as pd
from sklearn import preprocessing
import numpy as np
from utils.memory import DynamicExampleMemory


def batch_generator(X, Y, batch_size):
    """
    Sinh ngẫu nhiên các minibatch từ X, Y.
    Mỗi lần hết dữ liệu sẽ dừng (StopIteration).
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start+batch_size]
        yield X[batch_idx], Y[batch_idx]

def get_data():
    train_data = _load_data("data/UNSW_NB15_training-set.csv")
    test_data = _load_data("data/UNSW_NB15_testing-set.csv")

    train_data = _preprocess_data(train_data)
    test_data = _preprocess_data(test_data)

    X_train, Y_train = _separate_features_and_labels(train_data)
    X_test, Y_test = _separate_features_and_labels(test_data)

    return X_train, Y_train, X_test, Y_test

def _load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["id", "attack_cat"])
    return data

def _preprocess_data(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])

    min_max_scaler = preprocessing.MinMaxScaler()
    for column in data.columns:
        data[column] = min_max_scaler.fit_transform(data[column].values.reshape(-1,1))
    return data

def _separate_features_and_labels(data):
    Y = data.label.to_numpy()
    X = data.drop(columns="label").to_numpy()
    return X, Y

 #tiếp

def get_mixed_batch(train_gen, dem: DynamicExampleMemory,
                    new_bs: int, mem_bs: int,
                    X_full=None, Y_full=None):
    """
    train_gen : iterator sinh batch mới (batch_generator)
    dem       : instance DynamicExampleMemory
    new_bs    : kích thước batch mới
    mem_bs    : kích thước batch lấy từ memory
    X_full, Y_full : nếu train_gen hết StopIteration, sẽ tái khởi tạo new generator
    """
    try:
        x_new, y_new = next(train_gen)
    except StopIteration:
        # nếu iterator cũ hết, tạo lại
        train_gen = batch_generator(X_full, Y_full, new_bs)
        x_new, y_new = next(train_gen)

    # lấy từ memory
    x_mem, y_mem = dem.sample(mem_bs)

    # ghép batch mới + batch từ memory
    if len(x_mem) > 0:
        # X, Y là numpy arrays
        x_batch = np.vstack([x_new, np.array(x_mem)])
        y_batch = np.concatenate([y_new, np.array(y_mem)])
    else:
        x_batch, y_batch = x_new, y_new

    return x_batch, y_batch, x_new, y_new, train_gen
