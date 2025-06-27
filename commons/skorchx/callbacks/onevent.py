import skorch
from stdlib.tprint import tprint
from tqdm import tqdm


class OnEvent(skorch.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self._epoch = 0
        self._batch = 0
        self._count = 0
        self._tqdm = None

    def initialize(self):
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        tprint("begin train", force=True)
        pass

    def on_train_end(self, net, X=None, y=None, **kwargs):
        tprint("end train", force=True)
        pass

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self._count = len(dataset_train)//net.batch_size
        self._epoch += 1
        self._batch = 0
        # tprint(f"... begin epoch {self._epoch}", force=True, flush=True)
        self._tqdm_it = iter(tqdm(range(self._count)))
        pass

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # print("... end epoch")
        pass

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        # self._batch += 1
        # tprint(f"... ... batch {self._batch}")
        next(self._tqdm_it)
        pass

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        pass

    def on_grad_computed(self, net, named_parameters, X=None, y=None, training=None, **kwargs):
        pass

