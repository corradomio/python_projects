import matplotlib.pyplot as plt


def locations(x, y, closed=False, scatter=dict(), arrow=dict()):
    # plt.scatter(x[1:], y[1:], **scatter)
    # scatter["s"] = 10*scatter["s"]
    # scatter["c"] = "red"
    # plt.scatter(x[0:1], y[0:1], **scatter)
    plt.scatter(x, y, **scatter)
    # del scatter["s"]
    # plt.plot(x, y, **scatter)
# end


def locations_start(x, y, closed=False, scatter=dict(), arrow=dict()):
    scatter["c"] = "red"
    plt.scatter(x[0:1], y[0:1], **scatter)
# end


def arrows(x, y, closed=False, scatter=dict(), arrow=dict()):
    plt.scatter(x[1:], y[1:], **scatter)
    scatter["c"] = "red"
    plt.scatter(x[0:1], y[0:1], **scatter)
    n = len(x)
    for i in range(0,n-1):
        j = i+1
        plt.arrow(x[i],y[i], x[j]-x[i],y[j]-y[i], **arrow)
    if closed:
        plt.arrow(x[i], y[i], x[i] - x[0], y[i] - y[0], **arrow)

    # for i in range(0,n-1):
    #     plt.text(x[i], y[i], str(i))
# end
