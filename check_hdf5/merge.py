from path import Path as path
import h5py
from stdlib import picklex


ROOT = path("D:\\Projects.github\\article_projects\\article_causal_discovery_bool_data\\datasets")


def main():

    for o in [7,8,9,10]:
        s = str(o)

        p0 = ROOT / f"finfos-{o}-sampled-0.pickle"
        p5 = ROOT / f"finfos-{o}-sampled-5.pickle"

        c0 = picklex.load(p0)
        c5 = picklex.load(p5)

        c0s = c0[s]
        c5s = c5[s]
        c05 = c0s | c5s

        c0[s] = c05

        picklex.dump(c0, p0)
        pass
    pass


def main_hdf():

    for o in [8,9,10]:
        s = str(o)

        p0 = ROOT / f"dataset-{o}-sampled-0.hdf5"
        p5 = ROOT / f"dataset-{o}-sampled-5.hdf5"

        f0 = h5py.File(p0, "r+")
        f5 = h5py.File(p5, "r")

        c0 = f0[s]
        c5 = f5[s]

        for k in c5.keys():
            g5 = c5[k]
            n = g5.attrs['n']
            m = g5.attrs['m']
            wl_hash = g5.attrs['wl_hash']
            W = g5.attrs['adjacency_matrix']
            dataset = g5["dataset"]

            g0 = c0.create_group(k)
            g0.attrs['n'] = n
            g0.attrs['m'] = m
            g0.attrs['wl_hash'] = wl_hash
            g0.attrs['adjacency_matrix'] = W
            g0.create_dataset('dataset', dataset.shape,
                              dtype=dataset.dtype, data=dataset)

        f0.close()
        f5.close()
    pass


if __name__ == "__main__":
    main()
