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
        super(Callback, self).__init__()

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