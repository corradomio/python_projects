#
# errore nel calcolo di contact angle, comunque puo' essere corretto a partire d drop_angle
#
from path import Path as path
from stdlib.jsonx import load, dump
from stdlib.tprint import tprint
from joblib import Parallel, delayed

ROOT = path(r"E:\Datasets\WaterDrop\cropped")

def fix_contact_angle_error(jfile):
    # print(jfile)
    jdata = load(jfile)
    drop_angle = jdata['drop_angle']
    jdata['contact_angle'] = 90 - drop_angle
    dump(jdata, jfile)


def main():
    for idir in ROOT.dirs():
        tprint(idir, force=True)
        # for jfile in idir.files('*.json'):
        #     fix_contact_angle_error(jfile)
        Parallel(n_jobs=12)(delayed(fix_contact_angle_error)(ifile) for ifile in idir.files("*.json"))
    # end
    print("done")
# end


if __name__ == "__main__":
    main()
