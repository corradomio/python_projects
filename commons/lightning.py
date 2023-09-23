import numpy as np
import pytorch_lightning as pl
import torch
import random as rnd


# def fit(
#         self,
#         model: "pl.LightningModule",
#         train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
#         val_dataloaders: Optional[EVAL_DATALOADERS] = None,
#         datamodule: Optional[LightningDataModule] = None,
#         ckpt_path: Optional[str] = None,
#     ) -> None:

class NumpyDataloader:
    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 batch_size: int = 32,
                 shuffle: bool = True):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, (type(None), np.ndarray))
        assert y is None or len(X) == len(y)
        self.X: torch.Tensor = torch.from_numpy(X).float()
        self.y: torch.Tensor = torch.from_numpy(y).float()
        self.n: int = len(X)
        self._shuffle = shuffle
        self._indices = list(range(self.n))
        self.batch_size: int = batch_size
        self.at: int = 0

    def __len__(self):
        return (self.n + self.batch_size-1)//self.batch_size

    def __iter__(self):
        self.at = 0
        if self._shuffle:
            rnd.shuffle(self._indices)
        return self

    def __next__(self):
        if self.at >= self.n:
            raise StopIteration()
        bgn = self.at
        end = min(bgn+self.batch_size, self.n)
        self.at = end
        selected = self._indices[bgn:end]
        if self.y is None:
            return self.X[selected]
        else:
            return self.X[selected], self.y[selected]
    # end
# end


class LightningModule(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, y:np.ndarray, batch_size: int = 32, val=None, **kwargs):
        # is_cuda = next(self.parameters()).is_cuda
        trainer = pl.Trainer(**kwargs)
        train_dataloaders = NumpyDataloader(X, y, batch_size=batch_size)
        val_dataloaders = NumpyDataloader(*val, batch_size=batch_size) if isinstance(val, (list, tuple)) else None
        trainer.fit(self, train_dataloaders, val_dataloaders)
        return self

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.array:
        # float -> float | [float...]
        if isinstance(X, (int, float)):
            X = torch.tensor([X]).float().reshape(-1, 1)
        # [float...] -> [[float...] ...]
        elif isinstance(X, (list, tuple)):
            X = torch.tensor(X).float().reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        else:
            pass

        y = self.__call__(X)
        if y.shape == torch.Size([1,1]):
            return y.detach().numpy()[0, 0]
        elif y.shape[0] == 1:
            return y.detach().numpy()[0]
        else:
            return y.detach().numpy()
    # end

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def configure_trainer(self, **kwargs):
        trainer = pl.Trainer(**kwargs)
        return trainer
# end


class Module(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
# end