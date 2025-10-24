from math import pi
from joblib import Parallel, delayed

from commons import *
from stdlib.jsonx import load, dump
from stdlib.tprint import tprint


def sq(x): return x*x


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


#
#   0 -> background
#   1 -> water drop
#   2 -> table
#   3 -> dispenser
#
# Note: in the image the values used are [1-4]
#


def find_volume(jfile: path):
    try:
        tprint(f"... {jfile.stem}")

        jdata = load(jfile)
        drop_height = jdata["drop_height"]
        drop_radius = jdata["drop_radius"]

        drop_volume = pi*sq(drop_height)*(drop_radius - drop_height/3)
        jdata["drop_volume"] = drop_volume
        dump(jdata, jfile)
    except Exception as ex:
        print(jfile, ex)
        pass
# end


def main():
    # iroot = IMAGES_RESIZED
    iroot = IMAGES_ROOT
    for idir in iroot.dirs():
        # if idir.stem != "0000":
        #     continue
        tprint(idir, force=True)
        # for jfile in idir.files("drop_scene-*.json"):
        #     find_volume(jfile)
        Parallel(n_jobs=12)(delayed(find_volume)(jfile) for jfile in idir.files("drop_scene-*.json"))
    # end
    tprint("done", force=True)
# end



if __name__ == "__main__":
    main()
