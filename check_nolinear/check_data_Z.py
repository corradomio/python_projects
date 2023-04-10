import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning


class NonLinear4D(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        # self.l1 = nn.Linear(1, 8)
        # self.l2 = nn.Linear(8, 1)

        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),

            # nn.Linear(2, 32),
            # nn.ReLU(),
            # nn.Linear(32, 256),
            # nn.ReLU(),
            # nn.Linear(256, 32),
            # nn.ReLU(),
            # nn.Linear(32, 2),
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        y = self.layers(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss


def plot(df):
    ABCD_Z = df[['A', 'B', 'C', 'D', 'Z']]
    EF_Z = df[['E', 'F', 'Z']]

    ABCD_Z_0 = ABCD_Z[ABCD_Z['Z'] == 0]
    ABCD_Z_1 = ABCD_Z[ABCD_Z['Z'] == 1]

    EF_Z_0 = EF_Z[EF_Z['Z'] == 0]
    EF_Z_1 = EF_Z[EF_Z['Z'] == 1]

    plt.scatter(ABCD_Z_0['A'], ABCD_Z_0['B'])
    plt.scatter(ABCD_Z_1['A'], ABCD_Z_1['B'])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('A, B (Z)')
    plt.savefig("figures_z/ab_orig.png", dpi=300)
    plt.show()

    plt.scatter(ABCD_Z_0['C'], ABCD_Z_0['D'])
    plt.scatter(ABCD_Z_1['C'], ABCD_Z_1['D'])
    plt.xlabel('C')
    plt.ylabel('D')
    plt.title('C, D (Z)')
    plt.savefig("figures_z/cd_orig.png", dpi=300)
    plt.show()

    plt.scatter(EF_Z_0['E'], EF_Z_0['F'])
    plt.scatter(EF_Z_1['E'], EF_Z_1['F'])
    plt.xlabel('E')
    plt.ylabel('F')
    plt.title('E, F (Z)')
    plt.savefig("figures_z/ef_orig.png", dpi=300)
    plt.show()
    pass


def ab_cd(df):
    df_0 = df[df['Z'] == 0]
    df_1 = df[df['Z'] == 1]
    ab = df[['A', 'B']].to_numpy()
    cd = df[['C', 'D']].to_numpy()

    torch.set_float32_matmul_precision('medium')
    model = NonLinear4D()

    model.fit(ab, cd, batch_size=100, max_epochs=100)

    ab_0 = df_0[['A', 'B']].to_numpy()
    ab_1 = df_1[['A', 'B']].to_numpy()
    cd_0 = df_0[['C', 'D']].to_numpy()
    cd_1 = df_1[['C', 'D']].to_numpy()
    cd_0_hat = model.predict(ab_0)
    cd_1_hat = model.predict(ab_1)

    plt.scatter(ab_0[:, 0], ab_0[:, 1])
    plt.scatter(ab_1[:, 0], ab_1[:, 1])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('A, B (Z) Source')
    plt.savefig("figures_z/ab_cd_source.png", dpi=300)
    plt.show()

    plt.scatter(cd_0[:, 0], cd_0[:, 1])
    plt.scatter(cd_1[:, 0], cd_1[:, 1])
    plt.xlabel('C')
    plt.ylabel('D')
    plt.title('C, D (Z) Actual')
    plt.savefig("figures_z/ab_cd_actual.png", dpi=300)
    plt.show()

    plt.scatter(cd_0_hat[:, 0], cd_0_hat[:, 1])
    plt.scatter(cd_1_hat[:, 0], cd_1_hat[:, 1])
    plt.xlabel('C')
    plt.ylabel('D')
    plt.title('C, D (Z) Prediction')
    plt.savefig("figures_z/ab_cd_prediction.png", dpi=300)
    plt.show()

    pass


def cd_ab(df):
    df_0 = df[df['Z'] == 0]
    df_1 = df[df['Z'] == 1]
    ab = df[['A', 'B']].to_numpy()
    cd = df[['C', 'D']].to_numpy()

    torch.set_float32_matmul_precision('medium')
    model = NonLinear4D()

    model.fit(cd, ab, batch_size=100, max_epochs=400)

    ab_0 = df_0[['A', 'B']].to_numpy()
    ab_1 = df_1[['A', 'B']].to_numpy()
    cd_0 = df_0[['C', 'D']].to_numpy()
    cd_1 = df_1[['C', 'D']].to_numpy()

    ab_0_hat = model.predict(cd_0)
    ab_1_hat = model.predict(cd_1)

    plt.scatter(cd_0[:, 0], cd_0[:, 1])
    plt.scatter(cd_1[:, 0], cd_1[:, 1])
    plt.xlabel('C')
    plt.ylabel('D')
    plt.title('C, D (Z) Source')
    plt.savefig("figures_z/cd_ab_source.png", dpi=300)
    plt.show()

    plt.scatter(ab_0[:, 0], ab_0[:, 1])
    plt.scatter(ab_1[:, 0], ab_1[:, 1])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('A, B (Z) Actual')
    plt.savefig("figures_z/cd_ab_actual.png", dpi=300)
    plt.show()

    plt.scatter(ab_0_hat[:, 0], ab_0_hat[:, 1])
    plt.scatter(ab_1_hat[:, 0], ab_1_hat[:, 1])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('A, B (Z) Prediction')
    plt.savefig("figures_z/cd_ab_prediction.png", dpi=300)
    plt.show()


def cd_ef(df):
    df_0 = df[df['Z'] == 0]
    df_1 = df[df['Z'] == 1]
    cd = df[['C', 'D']].to_numpy()
    ab = df[['E', 'F']].to_numpy()

    torch.set_float32_matmul_precision('medium')
    model = NonLinear4D()

    model.fit(cd, ab, batch_size=100, max_epochs=400)

    cd_0 = df_0[['C', 'D']].to_numpy()
    cd_1 = df_1[['C', 'D']].to_numpy()
    ef_0 = df_0[['E', 'F']].to_numpy()
    ef_1 = df_1[['E', 'F']].to_numpy()

    ef_0_hat = model.predict(cd_0)
    ef_1_hat = model.predict(cd_1)

    plt.scatter(cd_0[:, 0], cd_0[:, 1])
    plt.scatter(cd_1[:, 0], cd_1[:, 1])
    plt.xlabel('C')
    plt.ylabel('D')
    plt.title('C, D (Z) Source')
    plt.savefig("figures_z/cd_ef_source.png", dpi=300)
    plt.show()

    plt.scatter(ef_0[:, 0], ef_0[:, 1])
    plt.scatter(ef_1[:, 0], ef_1[:, 1])
    plt.xlabel('E')
    plt.ylabel('F')
    plt.title('E, F (Z) Actual')
    plt.savefig("figures_z/cd_ef_actual.png", dpi=300)
    plt.show()

    plt.scatter(ef_0_hat[:, 0], ef_0_hat[:, 1])
    plt.scatter(ef_1_hat[:, 0], ef_1_hat[:, 1])
    plt.xlabel('E')
    plt.ylabel('F')
    plt.title('E, F (Z) Prediction')
    plt.savefig("figures_z/cd_ef_prediction.png", dpi=300)
    plt.show()



def main():
    df = pd.read_csv("data_con_XY.csv")

    plot(df)
    ab_cd(df)
    cd_ab(df)
    cd_ef(df)


if __name__ == "__main__":
    main()

