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

def plotHistory(metrics):
    num_metrics = len(metrics)
    plt.figure(figsize=(20, 4))

    for idx_metric, (metric_name, metric) in enumerate(metrics.items()):
        plt.subplot(1, num_metrics, idx_metric+1)

        plt.plot(metric)
        plt.title(metric_name)
        plt.xlabel('batch')

def visualization(baseline, prediction):
    plt.figure(figsize=(20, 10))
    n_images = len(baseline)

    for i in range(n_images):
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(baseline[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(prediction[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)