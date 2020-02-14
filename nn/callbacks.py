import numpy as np
import sys


class Callback(object):

    def on_train_begin(self, train_dataset, val_dataset=None):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, val_output=None, val_loss=None):
        pass

    def on_batch_begin(self, samples, labels):
        pass

    def on_batch_end(self, output, loss):
        pass


class ScoresCallback(Callback):

    def __init__(self):
        self.batches_count = None
        self.batch_labels = None
        self.val_dataset = None
        self.metrics = ['loss', 'accuracy']
        self.scores = {}
        self._total_epoch_scores = {}

    def on_train_begin(self, train_dataset, val_dataset=None):
        self.batches_count = 0
        if val_dataset is not None:
            self.val_dataset = val_dataset

    def on_epoch_begin(self):
        for metric in self.metrics:
            self.scores[metric] = []
            if self.val_dataset is not None:
                self.scores['val_' + metric] = []

    def on_epoch_end(self, val_output=None, val_loss=None):
        for metric in self.metrics:
            self.scores[metric] = self._total_epoch_scores[metric] / self.batches_count

        if (val_output and val_loss) is not None:
            _, val_labels = self.val_dataset
            self.scores['val_loss'] = val_loss
            self.scores['val_accuracy'] = compute_accuracy(val_labels, val_output)

    def on_batch_begin(self, samples, labels):
        self.batch_labels = labels
        for metric in self.metrics:
            self._total_epoch_scores[metric] = []

    def on_batch_end(self, output, loss):
        accuracy = compute_accuracy(self.batch_labels, output)
        self._total_epoch_scores['loss'].append(loss)
        self._total_epoch_scores['accuracy'].append(accuracy)
        self.batches_count += 1


class LoggingCallback(Callback):

    def on_train_begin(self, train_dataset, val_dataset=None):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, val_output=None, val_loss=None):
        pass

    def on_batch_begin(self, samples, labels):
        pass

    def on_batch_end(self, output, loss):
        sys.stdout.write('\r%s/%s - loss: %.4f - accuracy: %.4f'
                         % (len(X), len(X), self.scores['loss'][-1], self.scores['acc'][-1]))
        sys.stdout.write('\n')
        sys.stdout.flush()


def compute_accuracy(labels, outputs):
    return np.sum(labels == np.argmax(outputs, axis=1)) / outputs.shape[0]
