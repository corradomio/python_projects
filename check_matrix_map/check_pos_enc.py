import numpy as np
import matplotlib.pyplot as plt

#  P[k, 2i+0] = sin(k/n^(2i/d))
#  P[k, 2i+1] = cos(k/n^(2i/d))
#
# P[k, i+0] = sin(k/n^(i-0)/(2d))     0,2,4
# P[k, i+1] = cos(k/n^(i-1)/(2d))     1,3,5

# Note:
#   a = a[np.newaxis, :] is an alternative to a.reshape((1,) + a.shape)


# Positional encodings
def get_angles(pos, i, D, n=10000) -> np.ndarray:
    # pos: (n, 1)
    #   i: (1, m)
    angle_rates = 1 / np.power(n, (2 * (i // 2)) / np.float32(D))
    #   (n, 1) * (1, m) -> (n, m)
    return pos * angle_rates
# end


def positional_encoding_v1(seq_len, d, n=1000, dim=3) -> np.ndarray:

    # shape=(position, 1)
    pos_ = np.arange(seq_len)[:, np.newaxis]
    # shape=(1, D)
    i_ = np.arange(d)[np.newaxis, :]
    # shape=(position, D)
    angle_rads = get_angles(pos_, i_, d, n)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        # shape=(1, position, D)
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        # shape=(1, 1, position, D)
        pos_encoding = angle_rads[np.newaxis, np.newaxis,  ...]
    else:
        raise ValueError("Invalid dim")
    return pos_encoding
# end


def positional_encoding_v2(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i+0] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P[np.newaxis, ...]


def plot(pe):

    img = np.flip(pe[0].T, axis=0)

    plt.imshow(img)
    plt.show()



def main():

    pe1 = positional_encoding_v1(seq_len=20, d=10, n=50)
    plot(pe1)

    pe2 = positional_encoding_v2(seq_len=20, d=10, n=50)
    plot(pe2)

    pass


if __name__ == "__main__":
    main()
