# utils/memory.py
import random

class DynamicExampleMemory:
    def __init__(self, max_size):
        """
        max_size: số lượng mẫu tối đa lưu trong bộ nhớ
        """
        self.max_size = max_size
        # lưu dưới dạng list các dict {'x':…, 'y':…, 'loss':…}
        self.buffer = []

    def update(self, x_batch, y_batch, loss_batch):
        """
        x_batch: tensor hoặc array kích thước (B, …)
        y_batch: tensor hoặc array kích thước (B,)
        loss_batch: list hoặc tensor shape (B,) chứa loss từng sample
        """
        # thêm tất cả mẫu mới vào buffer
        for x, y, l in zip(x_batch, y_batch, loss_batch):
            self.buffer.append({'x': x, 'y': y, 'loss': float(l)})
        # giữ lại những mẫu có "độ quan trọng" cao nhất (theo loss)
        self.buffer.sort(key=lambda e: e['loss'], reverse=True)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[:self.max_size]

    def sample(self, batch_size):
        #trả về một minibatch gồm batch_size mẫu ngẫu nhiên từ buffer
        if len(self.buffer) == 0:
            return [], []
        batch_size = min(batch_size, len(self.buffer))
        chosen = random.sample(self.buffer, batch_size)
        xs = [e['x'] for e in chosen]
        ys = [e['y'] for e in chosen]
        return xs, ys
