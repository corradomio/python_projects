from stdlib.tprint import tprint
import arrayfire as af
import arrayfire.random as afr


def calc_pi_device(samples):
    # Simple, array based API
    # Generate uniformly distributed random numers
    x = af.randu(samples)
    y = af.randu(samples)
    # Supports Just In Time Compilation
    # The following line generates a single kernel
    within_unit_circle = (x * x + y * y) < 1
    # Intuitive function names
    return 4 * af.count(within_unit_circle) / samples

def main():
    # 'default' 'unified' 'cpu' 'cuda' 'opencl'
    af.set_backend("cuda")
    # engine = af.random.Random_Engine()
    #
    # print(afr.randn(10, engine=engine))

    tprint("start")
    calc_pi_device(10000000)
    tprint("done")
    pass


if __name__ == "__main__":
    main()
