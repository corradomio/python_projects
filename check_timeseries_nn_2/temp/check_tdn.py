#
# https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks
#
# ======================
#     Dataset Utils
# ======================
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchx.nn import TemporalConvNetwork


def get_dataset(x, y):
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_ndarray(embedding_values):
    results = []
    for row in embedding_values:
        arr = np.array(row)
        results.append(
            np.pad(arr, ((10 - arr.shape[0], 0), (0, 0)), 'constant')
        )
    # shape: (examples, emb_dim, seq_length)
    return np.transpose(np.stack(results), (0, 2, 1))


def read_dataset(data_dir=Path("data/")):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    df_train = pd.read_json(data_dir / 'train.json')
    df_test = pd.read_json(data_dir / 'test.json')
    x_train = get_ndarray(df_train.audio_embedding)
    y_train = df_train.is_turkey.values
    x_test = get_ndarray(df_test.audio_embedding)
    test_id = df_test.vid_id
    return x_train, y_train, x_test, test_id


# ===============================================
#     Model Creation, Training, and Inference
# ==============================================
# helperbot:
#   python-telegram-bot
#
import logging

from helperbot.bot import BaseBot
from helperbot.lr_scheduler import TriangularLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import torch.nn as nn
import pandas as pd
from pathlib import Path


class TurkeyBot(BaseBot):
    name = "Turkey"

    def __init__(self, model, train_loader, val_loader, *, optimizer,
                 avg_window=20, log_dir=Path("./cache/logs/"),
                 log_level=logging.INFO, checkpoint_dir=Path("./cache/model_cache/")):
        super().__init__(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, avg_window=avg_window,
            log_dir=log_dir, log_level=log_level, checkpoint_dir=checkpoint_dir,
            batch_idx=0, echo=False, device=DEVICE,
            criterion=torch.nn.BCEWithLogitsLoss()
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_format = "%.8f"


class TCNModel(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNetwork(
            128, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))


DEVICE = "cpu"


def main():
    x_train, y_train, x_test, test_id = read_dataset("../input/")

    test_loader = get_dataloader(
        x_test, np.zeros(x_test.shape[0]), batch_size=128, shuffle=False)

    test_pred_list, val_losses = [], []
    kf = StratifiedKFold(n_splits=8, random_state=31829, shuffle=True)
    for train_index, valid_index in kf.split(x_train, y_train):
        train_loader = get_dataloader(
            x_train[train_index], y_train[train_index],
            batch_size=32, shuffle=True
        )
        val_loader = get_dataloader(
            x_train[valid_index], y_train[valid_index],
            batch_size=128, shuffle=False
        )

        model = TCNModel(num_channels=[20] * 2, kernel_size=3, dropout=0.25)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
        # optimizer = WeightDecayOptimizerWrapper(
        #     optimizer, weight_decay=5e-3
        # )
        batches_per_epoch = len(train_loader)
        bot = TurkeyBot(
            model, train_loader, val_loader,
            optimizer=optimizer, avg_window=batches_per_epoch
        )
        n_steps = batches_per_epoch * 20
        scheduler = TriangularLR(
            optimizer, max_mul=8, ratio=9,
            steps_per_cycle=n_steps
        )
        bot.train(
            n_steps,
            log_interval=batches_per_epoch // 2,
            snapshot_interval=batches_per_epoch,
            early_stopping_cnt=10, scheduler=scheduler)
        val_preds = torch.sigmoid(bot.predict_avg(
            val_loader, k=3, is_test=True).cpu()).numpy().clip(1e-5, 1 - 1e-5)
        loss = log_loss(y_train[valid_index], val_preds)
        if loss > 0.2:
            # Ditch folds that perform terribly
            bot.remove_checkpoints(keep=0)
            continue
        print("AUC: %.6f" % roc_auc_score(y_train[valid_index], val_preds))
        print("Val loss: %.6f" % loss)
        val_losses.append(loss)
        test_pred_list.append(torch.sigmoid(bot.predict_avg(
            test_loader, k=3, is_test=True).cpu()).numpy().clip(1e-5, 1 - 1e-5))
        bot.remove_checkpoints(keep=0)

    val_loss = np.mean(val_losses)
    test_preds = np.mean(test_pred_list, axis=0)
    print("Validation losses: %.6f +- %.6f" %
          (np.mean(val_losses), np.std(val_losses)))

    df_sub = pd.DataFrame({
        "vid_id": test_id,
        "is_turkey": test_preds
    })
    df_sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
