import os
import tensorflow as tf
import flwr as fl
import utils.data_loader as data_loader
import utils.model_loader as model_loader
from utils.memory import DynamicExampleMemory
from utils.data_loader import batch_generator, get_mixed_batch


# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = model_loader.get_model(self.X_train.shape[1:])
        # --- thêm vào ---
        self.dem = DynamicExampleMemory(max_size=5000)        # hoặc bất cứ giá trị phù hợp
        self.train_gen = batch_generator(self.X_train, self.Y_train, batch_size=32)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # 1. load weights từ server
        self.model.set_weights(parameters)

        # 2. vòng local epochs (ở bài báo mỗi client train 1 epoch)
        total_steps = len(self.X_train) // 32
        for _ in range(total_steps):
            # 2.1 lấy batch hỗn hợp và batch mới tách riêng
            x_batch, y_batch, x_new, y_new, self.train_gen = get_mixed_batch(
                self.train_gen,
                self.dem,
                new_bs=32,
                mem_bs=16,
                X_full=self.X_train,
                Y_full=self.Y_train
            )

            # Chuyển x_new, y_new về Tensor nếu còn là numpy array
            x_new_t = tf.convert_to_tensor(x_new)
            y_new_t = tf.convert_to_tensor(y_new)

            # 2.2 forward trên batch mới để tính loss cá thể
            logits = self.model(x_new_t, training=True)
            losses_new = tf.keras.losses.sparse_categorical_crossentropy(
                y_new_t,        # labels
                logits,         # model outputs
                from_logits=False  # nếu model dùng softmax ở cuối
            )  # losses_new là tensor shape (new_bs,)
            # 2.3 update memory với loss của mẫu mới
            self.dem.update(x_new.tolist(), y_new.tolist(),
                            losses_new.numpy() if isinstance(losses_new, tf.Tensor) else losses_new)

            # 2.4 train chung batch (new + memory)
            self.model.train_on_batch(x_batch, y_batch)

        # 3. trả lại tham số, số mẫu và metrics
        weights = self.model.get_weights()
        # bạn có thể thu thập loss/acc cuối cùng bằng evaluate trên X_train, hoặc track thủ công
        train_loss, train_acc = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        return weights, len(self.X_train), {"loss": train_loss, "accuracy": train_acc}

    def evaluate(self, parameters, _):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_numpy_client(server_address=server_address, client=Client())