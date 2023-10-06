import skorch.callbacks as skc
from copy import deepcopy


class EarlyStopping(skc.EarlyStopping):

    def __init__(self, *args, min_epochs=0, **kwargs):
        """
        Extends skorch.callbacks.EarlyStopping adding the parameter 'min_epochs':
        the train proceeds for a minimum number of epochs

        :param min_epochs: minimum number of epochs to use in training
        """
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1, "epoch"]
        if epoch < self.min_epochs:
            return

        current_score = net.history[-1, self.monitor]

        if self._is_over_threshold(current_score):
            return

        super().on_epoch_end(net, **kwargs)

    def _is_over_threshold(self, current_score):
        if self.lower_is_better:
            if current_score > self.dynamic_threshold_ + self.threshold:
                self.dynamic_threshold_ = current_score
                self.misses_ = 0
                return True
        else:
            if current_score < self.dynamic_threshold_ - self.threshold:
                self.dynamic_threshold_ = current_score
                self.misses_ = 0
                return True
        return False
# end
