from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import Callback

class LRFinder(Callback):
    """Finds optimal learning rate"""
    def __init__(self, steps_per_epoch, min_lr=1e-6, max_lr=1e-1, **kwargs):
        super().__init__(**kwargs)

        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.lr_factor = (self.max_lr / self.min_lr) ** (1. / self.steps_per_epoch)
        self.iteration = 0
        self.history = {}

    def on_train_begin(self, logs=None):
        # Begin training, set minimum learning rate
        K.set_value(self.model.optimizer.lr, self.min_lr)


    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        logs = logs or {}

        # Get current learning rate
        lr = K.get_value(self.model.optimizer.lr)

        # Update learning rate
        lr *= self.lr_factor
        K.set_value(self.model.optimizer.lr, lr)

        # Logs
        self.history.setdefault('lr', []).append(lr)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_train_end(self, logs=None):
        self.plotLoss()

    def plotLoss(self):
        # Plot loss as function of learning rate (log scale)
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


class LRFinderAuto(LRFinder):
    """Finds optimal learning rate automaticly"""
    def __init__(self, steps_per_epoch, *args, **kwargs):
        super().__init__(steps_per_epoch, *args, **kwargs)
        
        # Smother loss curve
        self.smooth_loss = None

        # Maximum learning rate
        self.maxLR = None

        # Optimization for first epoch
        self.opti = True

    def on_train_begin(self, logs=None):
        # Back up weights
        self.init_weights = self.model.get_weights()

        super().on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):
        # If during optimization epoch
        if self.opti:
        	super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # End of the optimization epoch
        if epoch == 0:
            self.lr = self.history['lr']
            self.loss = self.history['loss']

            # Stop optimization
            self.opti = False
            
            # Set optimal learning rate
            self.autoChooseLearningRate()
            K.set_value(self.model.optimizer.lr, self.maxLR)
            
            # Reset weights
            self.model.set_weights(self.init_weights)
    
    def autoChooseLearningRate(self):
        from scipy.signal import savgol_filter

        # Experimental, to be replaced
        window_length = len(self.loss) // 10
        window_length += (not window_length % 2)
        self.smooth_loss = savgol_filter(self.loss, window_length=window_length, polyorder=2)
        self.maxLR = self.lr[np.argmin(self.smooth_loss)]

        print(f"Maximal learning rate: {self.maxLR}")