import skorch


class EarlyStopping(skorch.callbacks.EarlyStopping):

    def __init__(self, min_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs
        self._epoch = 0

    def on_epoch_end(self, net, **kwargs):
        self._epoch += 1
        if self._epoch >= self.min_epochs:
            super().on_epoch_end(net, **kwargs)
    # end
# end
