import numpy as np
import matplotlib.pyplot as plt


def model1():
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)

    x, y = np.meshgrid(x, y)

    x_0 = x > 0
    x_1 = x < 0
    y_0 = y > 0
    y_1 = y < 0

    plt.scatter(x[x_0], y[x_0])
    plt.scatter(x[x_1], y[x_1])
    plt.show()

    plt.scatter(x[y_0], y[y_0])
    plt.scatter(x[y_1], y[y_1])
    plt.show()

    ro = np.sqrt(np.power(x, 2) + np.power(y, 2))
    theta = np.arctan2(x, y)

    plt.scatter(theta[x_0], ro[x_0])
    plt.scatter(theta[x_1], ro[x_1])
    plt.show()

    plt.scatter(theta[y_0], ro[y_0])
    plt.scatter(theta[y_1], ro[y_1])
    plt.show()

    xy_00 = x_0 * y_0
    xy_01 = x_0 * y_1
    xy_10 = x_1 * y_0
    xy_11 = x_1 * y_1

    plt.scatter(x[xy_00], y[xy_00])
    plt.scatter(x[xy_01], y[xy_01])
    plt.scatter(x[xy_10], y[xy_10])
    plt.scatter(x[xy_11], y[xy_11])
    plt.show()

    plt.scatter(theta[xy_00], ro[xy_00])
    plt.scatter(theta[xy_01], ro[xy_01])
    plt.scatter(theta[xy_10], ro[xy_10])
    plt.scatter(theta[xy_11], ro[xy_11])
    plt.show()

    pass


def model2():
    npi = np.pi

    ro = np.arange(0, 1, 0.01)
    theta = np.arange(0, 2*npi, 0.01)

    ro, theta = np.meshgrid(ro, theta)

    ro_0 = ro > 0.5
    ro_1 = ro < 0.5
    theta_0 = theta > npi
    theta_1 = theta < npi

    plt.scatter(ro[ro_0], theta[ro_0])
    plt.scatter(ro[ro_1], theta[ro_1])
    plt.show()

    plt.scatter(ro[theta_0], theta[theta_0])
    plt.scatter(ro[theta_1], theta[theta_1])
    plt.show()

    x = ro*np.cos(theta)
    y = ro*np.sin(theta)

    plt.scatter(x[ro_0], y[ro_0])
    plt.scatter(x[ro_1], y[ro_1])
    plt.show()

    plt.scatter(x[theta_0], y[theta_0])
    plt.scatter(x[theta_1], y[theta_1])
    plt.show()

    rt_00 = ro_0 * theta_0
    rt_01 = ro_0 * theta_1
    rt_11 = ro_1 * theta_1
    rt_10 = ro_1 * theta_0

    plt.scatter(ro[rt_00], theta[rt_00])
    plt.scatter(ro[rt_01], theta[rt_01])
    plt.scatter(ro[rt_11], theta[rt_11])
    plt.scatter(ro[rt_10], theta[rt_10])
    plt.show()

    plt.scatter(x[rt_00], y[rt_00])
    plt.scatter(x[rt_01], y[rt_01])
    plt.scatter(x[rt_11], y[rt_11])
    plt.scatter(x[rt_10], y[rt_10])
    plt.show()

    pass


def main():
    model1()
    # model2()


if __name__ == "__main__":
    main()
