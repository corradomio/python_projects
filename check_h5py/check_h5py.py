import h5py
from stdlib.is_instance import is_instance



def main():
    f: h5py.Group = h5py.File('dataset-4-0.hdf5', 'r')
    assert is_instance(f, h5py.Group)
    print(f.keys())
    for k in f.keys():
        f1: h5py.Group = f.get(k)
        assert is_instance(f1, h5py.Group)
        print("...", f1.keys())
        for k1 in f1.keys():
            f2: h5py.Dataset = f1.get(k1)
            assert is_instance(f2, h5py.Group)
            print("... ...", f2.keys())
            for k2 in f2.keys():
                ds: h5py.Dataset = f2.get(k2)
                assert is_instance(ds, h5py.Dataset)
                print("... ... ...", ds.attrs.keys())




if __name__ == "__main__":
    main()
