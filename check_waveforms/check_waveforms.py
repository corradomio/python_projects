from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from waveforms import *

def plot(name, x, y):
    plt.clf()
    plt.plot(x,y)
    plt.savefig("plots/{}.png".format(name), dpi=300)



def main():
    x = np.arange(0,3,.01,dtype=float)

    plot("sin_wave", x, sin_wave(x))
    plot("sin_wave_2", x, sin_wave(x, 0.25))
    plot("sinabs_wave", x, sinabs_wave(x))
    plot("cos_wave", x, cos_wave(x))
    plot("sawtooth_wave", x, sawtooth_wave(x))
    plot("inverted_sawtooth_wave", x, inverted_sawtooth_wave(x))
    plot("square_wave", x, square_wave(x))
    plot("triangle_wave", x, triangle_wave(x))
    plot("sintooth_wave", x, sintooth_wave(x))
    plot("triangletooth_wave", x, triangletooth_wave(x))

    plot("fourier_wave", x, fourier_wave(0,[1,-.5], [0, .3],x))
    plot("hadamard_wave", x, hadamard_wave(0, [1,-.5], [0, .3], x))
    plot("slant_haar_wave", x, slant_haar_wave(0, [1,-.5], [0, .3], x))

    k=10
    a: np.ndarray = cast(np.ndarray, np.random.uniform(size=k))
    p: np.ndarray = cast(np.ndarray, np.random.uniform(size=k))

    plot("fourier_wave_k", x, fourier_wave(0,a,p,x))
    plot("hadamard_wave_k", x, hadamard_wave(0,a,p,x))
    plot("slant_haar_wave_k", x, slant_haar_wave(0,a,p,x))
    pass

if __name__ == "__main__":
    main()