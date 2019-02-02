import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback

class LossHistory(Callback):
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = ([], [])

    def on_epoch_end(self, epoch, logs=None):
        for key, metric in self.metrics.items():
            self.metrics[key][0].append( logs.get(key) )
            self.metrics[key][1].append( logs.get('val_' + key) )


def plotHistory(metrics):
    num_metrics = len(metrics)
    plt.figure(figsize=(20, 4))
    for idx_metric, (metric_name, (metric, metric_val)) in enumerate(metrics.items()):
        plt.subplot(1, num_metrics, idx_metric+1)

        plt.plot(metric[1:], '#EF6C00', label='train')
        plt.plot(metric_val[1:], '#0077BB', label='eval')
        plt.title(metric_name)
        plt.xlabel('epoch')
    plt.legend()

def visualization_train(dataset, netout):
    images, labels = dataset
    reconstructions, predictions = netout

    plt.figure(figsize=(20, 10))
    n_images = len(images)

    for i in range(n_images):
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(images[i], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(labels[i])

        ax = plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(reconstructions[i], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(predictions[i])
        

def visualization_data(images, labels, predictions):

    plt.figure(figsize=(20, 10))
    n_images = len(images)

    for i in range(n_images):
        ax = plt.subplot(1, n_images, i + 1)
        plt.imshow(images[i], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f'{predictions[i]} ({labels[i]})')
        
def rotation_accuracy(model, dataGenerator, m_test, batch_size=32, n_points=10):
    rotations = np.linspace(0, 180, n_points)
    data_augmentation = {'rotation_range': 0}
    
    test_accuracy = []
    for deg in rotations:
        # Select data
        data_augmentation['rotation_range'] = deg
        testGenerator = dataGenerator('test', batch_size=batch_size, reshape=False, **data_augmentation)

        # Test accuracy
        test_acc = model.evaluate_generator(testGenerator, steps=m_test//batch_size)[1]
        print(f'Test acc [{deg}°]:\t{round(test_acc, 3)}')
        test_accuracy.append(test_acc)
        
    # Plot
    plt.plot(rotations, test_accuracy)
    plt.title('rotation accuracy')
    plt.xlabel('rotation (deg)')
    plt.ylabel('test accuracy')