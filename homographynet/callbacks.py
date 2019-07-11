#coding=utf-8
import tensorflow as tf
import math
from matplotlib import pyplot as plt

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.
    See Caffe SGD docs
    """
    def __init__(self, base_lr, gamma, step_size):
        super().__init__()
        self._base_lr = base_lr
        self._gamma = gamma
        self._step_size = step_size # drop the learning rate every xx iterations
        self._steps = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._steps = epoch * self.params['steps']

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._steps % self._step_size == 0:
            exp = int(self._steps / self._step_size)
            lr = self._base_lr * (self._gamma ** exp)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            print('New learning rate:', lr)


class LearningRateScheduler_CosineDecay(tf.keras.callbacks.Callback):
    """Cosine decay learning rate scheduler.
    """
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self._base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._steps = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._steps = epoch * self.params['steps']

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._steps < self.warmup_steps:
            lr = float(self._steps) * self._base_lr / self.warmup_steps
        else:
            lr = self._base_lr * 0.5 * (1 + math.cos( float(self._steps - self.warmup_steps) * math.pi / (self.total_steps - self.warmup_steps)))

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(' - learning rate: {:.6f}'.format(lr), end='')


def test_cosine_decay(base_lr, warmup_steps, total_steps):
    learning_rates = []
    for _steps in range(1, 1+total_steps):
        if _steps < warmup_steps:
            lr = float(_steps) * base_lr / warmup_steps
        else:
            lr = base_lr * 0.5 * (1 + math.cos( float(_steps - warmup_steps) * math.pi / (total_steps - warmup_steps)))
        learning_rates.append(lr)
    print(learning_rates[-1])
    plt.plot(list(range(len(learning_rates))), learning_rates)
    plt.show()

if __name__ == '__main__':
    test_cosine_decay(0.1, 100, 1000)
