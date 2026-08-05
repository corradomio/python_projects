from math import radians, atan2

import matplotlib.pyplot as plt
from droplet import Droplet, droplet_radius, droplet_base, droplet_radius_angle
from stdlib.tprint import tprint


def main2():
    dl = Droplet(2, 5)
    pts = dl.points(30)

    plt.plot(pts[:,0], pts[:,1])
    plt.gca().set_aspect("equal")
    plt.show()

    dl = Droplet(2, 10)
    pts = dl.points(30)

    plt.plot(pts[:,0], pts[:,1])
    plt.gca().set_aspect("equal")
    plt.show()

    dl = Droplet(2, 20)
    pts = dl.points(30)

    plt.plot(pts[:,0], pts[:,1])
    plt.gca().set_aspect("equal")
    plt.show()




def main3():
    # liquid_drop(2, 1, 45, 10, 5)
    # liquid_drop(2, 1, 45, 80, 40)
    # liquid_drop(2, 1, 90, 80, 40)
    # liquid_drop(2, 1, (45,90), 80, 40)

    # dl = Droplet(2, (45, 45))
    # dl.save_obj("droplets", 10,20)
    # dl.save_obj("droplets", 20,40)
    # dl.save_obj("droplets", 30,60)
    # dl.save_obj("droplets", 40,80)

    tprint(f"Start ...", force=True)

    sangle = 2
    eangle = 135+1
    dangle = 2
    for langle in range(sangle, eangle, dangle):
        for rangle in range(sangle, eangle, dangle):
            tprint(f"{langle:3} x {rangle:3}")
            dl = Droplet(1, (langle, rangle))
            dl.save_obj("droplets", 30, 60)
    # end

    ANGLES = [30,60,90,120,135]
    for langle in ANGLES:
        for rangle in ANGLES:
            tprint(f"{langle:3} x {rangle:3}")
            dl = Droplet(1, (langle, rangle))
            dl.save_obj("droplets", 30, 60)
    pass



def main4():
    tprint(f"Start ...", force=True)

    for a in range(2,178):
        langle = a
        rangle = a
        tprint(f"{langle:3} x {rangle:3}")
        dl = Droplet(1, (langle, rangle))
        dl.save_obj("droplets", 30, 60)


def main5():
    print(droplet_radius(4, radians(90)))
    print(droplet_radius(4 * 8, radians(90)) / 2)
    print(droplet_radius(4, radians(45)))

    print(droplet_base(4,radians(90)))
    print(droplet_base(4,radians(180)))
    print(droplet_base(4,radians(45)))


def main():
    print(droplet_radius_angle(2, 1))
    print(droplet_radius_angle(1, 1))


if __name__ == "__main__":
    main()
