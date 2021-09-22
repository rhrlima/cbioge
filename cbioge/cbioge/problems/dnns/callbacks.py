import datetime as dt
from keras.callbacks import Callback


class TimedStopping(Callback):
    ''' Stop training when enough time has passed.
        Verification is made at each batch end.
        
        # Arguments
        seconds: maximum time before stopping
        verbose: verbosity mode
    '''

    def __init__(self, seconds=None, verbose=0):
        super().__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = dt.datetime.today()

    def on_batch_end(self, epoch, logs=None):
        if dt.datetime.today() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)


class EpochReport(Callback):

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
    
    def on_epoch_end(self, epoch, logs):

        if self.epochs > 0 and epoch % self.epochs == 0:
            text = f'{epoch} - loss {logs["loss"]} - acc {logs["acc"]}'
            if 'val_loss' in logs:
                text += f' val_loss {logs["val_loss"]} - val_acc {logs["val_acc"]}'
            print(text)