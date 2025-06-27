from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from stdlib.jsonx import load
from stdlib.dictx import  dict_get
from utils import *


def main():
    D = np.random.normal(0, 1, 1000)
    for i in range(20):
        D[i] = .99
    # D = np.random.exponential(1, 1000)
    plt.boxplot(
        D,
        # positions=[1,2,3,4,5],
        positions=[1],
        showmeans=True, meanline=True,
        notch=False,
        # widths=0.3
        vert=False
    )
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()

    pass
# end



if __name__ == "__main__":
    main()
