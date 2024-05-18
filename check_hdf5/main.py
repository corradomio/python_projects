import numpy as np
import h5py


def main():
    print("load ...")
    f = h5py.File(r"D:\Mathematica\UniverseSim\quijote\fiducial\molino.z0.0.fiducial.nbody0.hod0.hdf5", 'r')
    print(f.keys())
    pos: np.ndarray = f.get('pos')[:]
    vel: np.ndarray = f.get('vel')[:]
    print("... done")
    np.savetxt('pos.csv', pos, header='x,y,z')
    np.savetxt('vel.csv', vel, header='vx,vy,vz')
# end


if __name__ == "__main__":
    main()
