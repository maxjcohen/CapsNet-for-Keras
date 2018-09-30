from matplotlib import pyplot as plt
from keras.callbacks import Callback

class LossHistory(Callback):
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = []

    def on_batch_end(self, batch, logs={}):
        for key, metric in self.metrics.items():
            self.metrics[key].append( logs.get(key) )

def plotHistory(loss, acc):
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 2, 1)

    plt.plot(loss)
    plt.title('loss through training')
    plt.ylabel('loss')
    plt.xlabel('batch')

    plt.subplot(1, 2, 2)
    plt.plot(acc, 'g')
    plt.title('accuracy through training')
    plt.ylabel('accuracy')
    plt.xlabel('batch')
