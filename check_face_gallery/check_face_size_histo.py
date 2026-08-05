from pathlib import Path
from typing import cast

from stdlib.tprint import tprint
from stdlib import loggingx as logging
from stdlib.collections import flatten
from PIL import Image
from matplotlib import pyplot as plt
from joblibx import Parallel, delayed


def analyze_faces_track_dir(i, n, track_dir):
    if not track_dir.is_dir(): return [],[]

    # tprint(f"... {track_dir.name} ({i + 1:4}/{n})", force=False)

    face_dir = track_dir / "face"
    if not face_dir.exists(): return [],[]

    w_track = []
    h_track = []

    for face_file in face_dir.iterdir():
        if not face_file.name.endswith(".jpg"): continue

        image = Image.open(face_file)
        W, H = image.size
        image.close()

        w_track.append(W)
        h_track.append(H)
        pass
    # end

    return w_track, h_track
# end


def analyze_faces_tracks_root(r, R, tracks_root):
    if not tracks_root.is_dir(): return [], []
    if (tracks_root / "impurity.json").exists(): return [], []
    if (tracks_root / "segmented.json").exists(): return [], []

    tprint(f"Analyze {tracks_root} ({r + 1:2}/{R}) ...")

    track_dirs = [track_dir for track_dir in tracks_root.iterdir()]
    n = len(track_dirs)

    # w_tracks = []
    # h_tracks = []
    #
    # for i, track_dir in enumerate(track_dirs):
    #     w_track, h_track = analyze_faces_track_dir(i, n, track_dir)
    #
    #     w_tracks += w_track
    #     h_tracks += h_track
    # # end

    wh_tracks:list[tuple[list[int], list[int]]] = cast(list[tuple[list[int], list[int]]],
    Parallel(n_jobs=14)(
        delayed(analyze_faces_track_dir)(i, n, track_dir)
        for i, track_dir in enumerate(track_dirs)
    ))

    # print(wh_tracks)

    w_tracks: list = []
    h_tracks: list = []
    for wh_track in wh_tracks:
        w_tracks += wh_track[0]
        h_tracks += wh_track[1]

    if len(w_tracks) < 10:
        return w_tracks, h_tracks

    n_faces = len(w_tracks)

    plt.clf()
    plt.title(f"{tracks_root.name}: faces={n_faces}")
    plt.hist(w_tracks, bins=100, label="W", alpha=0.5)
    plt.hist(h_tracks, bins=100, label="H", alpha=0.5)
    plt.legend()
    plt.xlim(0, 300)
    plt.savefig(f"plots/{tracks_root.name}.png", dpi=300)

    return w_tracks, h_tracks
# end


def analyze_faces_folder(root: Path):

    w_list = []
    h_list = []

    tracks_roots = [tracks_root for tracks_root in root.iterdir()]
    R = len(tracks_roots)

    for r, tracks_root in enumerate(tracks_roots):
        w_tracks, h_tracks = analyze_faces_tracks_root(r, R, tracks_root)

        w_list += w_tracks
        h_list += h_tracks
    # end



    n_faces = len(w_list)

    plt.clf()
    plt.title(f"root-{root.name}: faces={n_faces}")
    plt.hist(w_list, bins=100, label="W", alpha=0.5)
    plt.hist(h_list, bins=100, label="H", alpha=0.5)
    plt.legend()
    plt.xlim(0, 300)
    plt.savefig(f"plots/root-{root.name}.png", dpi=300)

    return w_list, h_list



def main():
    roots = [
        # Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_mini"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_2"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_1"),
    ]

    w_all = []
    h_all = []
    for root in roots:
        w_list, h_list = analyze_faces_folder(root)
        w_all += w_list
        h_all += h_list


    plt.clf()
    plt.hist(w_all, bins=100, label="W")
    plt.hist(h_all, bins=100, label="H")
    plt.legend()
    plt.savefig("plots/histograms.png", dpi=300)


    tprint("Done")


if __name__ == '__main__':
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    main()

