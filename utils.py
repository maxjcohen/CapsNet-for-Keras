from matplotlib import pyplot as plt
from keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

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
