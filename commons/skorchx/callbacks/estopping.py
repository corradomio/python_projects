import skorch


class EarlyStopping(skorch.callbacks.EarlyStopping):

    def __init__(self, warmup=0, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup
        self._epoch = 0

    def on_epoch_end(self, net, **kwargs):
        self._epoch += 1
        if self._epoch >= self.warmup:
            super().on_epoch_end(net, **kwargs)
    # end
# end
