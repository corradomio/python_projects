import numpy as np
import h5py


def main():
    print("load ...")
    for o in [7,8,9,10]:
        for i in range(6):
            f = h5py.File(f"D:\\Projects.github\\article_projects\\article_causal_discovery_bool_data\\datasets\\dataset-{o}-sampled-{i}.hdf5", 'r')
            g = f[str(o)]
            print(o,i,len(g.keys()))
# end


if __name__ == "__main__":
    main()
