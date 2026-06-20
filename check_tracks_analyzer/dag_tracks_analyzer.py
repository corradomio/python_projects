import datetime as dt
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Self, cast

import networkx as nx

import netx
from stdlib import jsonx
from stdlib import loggingx as logging
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
from stdlib.sortedx import sort_by_key

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

JSON = dict[str, Any]

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def iop_ratio(bbox1: list[int], bbox2: list[int]) -> float:
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    assert x11 < x12 and y11 < y12
    assert x21 < x22 and y21 < y22

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    xu1 = min(x11, x21)
    yu1 = min(y11, y21)
    xu2 = max(x12, x22)
    yu2 = max(y12, y22)

    iop_ratio = ((xi2 - xi1) * (yi2 - yi1)) / ((xu2 - xu1) * (yu2 - yu1))
    return iop_ratio


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Track:
    # {
    #     "camera_id": 0,
    #     "person_id": 165,
    #     "present_start": "2026-03-26 12:07:14",
    #     "present_end": "2026-03-26 12:07:19",
    #
    #     "start_person_box": [ 0, 766, 189, 1080 ],
    #     "end_person_box": [ 0, 706, 294, 1080 ]
    #     "more_persons": false,
    #
    #     "present_frames": 54,
    #     "person_box": [ 0, 721, 336, 1080],
    #     "person_box_no_margin": [ 1, 768, 293, 1077 ],
    #     "not_dress_well": true,
    # }

    def __init__(self, track_path:Path, meta: JSON):
        self._path: Path = track_path
        # self._meta = meta
        self._name: str = track_path.name
        self._camera = meta["camera_id"]
        self._track = meta["person_id"]
        self._start_bbox: list[int] = meta["start_person_box"]
        self._end_bbox: list[int] = meta["end_person_box"]
        self._start_time: datetime = datetime.strptime(meta["present_start"], DATETIME_FORMAT)
        self._end_time: datetime = datetime.strptime(meta["present_end"], DATETIME_FORMAT)

    @property
    def name(self) -> str:
        # track name: <camera>_<track>_DONE
        return self._name

    @property
    def camera(self) -> int:
        return self._camera

    @property
    def track(self) -> int:
        return self._track

    @property
    def day(self) -> date:
        # day of the track (yyy-mm-dd)
        return self._start_time.date()

    @property
    def duration(self) -> int:
        # time duration of the track
        return (self._end_time - self._start_time).seconds

    @property
    def is_random_crop_only(self) -> bool:
        # contains ONLY "random_crop"
        n_dirs = sum(1 for item in self._path.iterdir() if item.is_dir())
        return n_dirs <= 1

    @property
    def start_time(self) -> date:
        return self._start_time

    @property
    def end_time(self) -> date:
        return self._end_time

    @property
    def start_bbox(self) -> list[int]:
        return self._start_bbox

    @property
    def end_bbox(self) -> list[int]:
        return self._end_bbox

    def is_before(self, that: Self) -> bool:
        # [self] [that]
        return self._end_time <= that._start_time

    def is_after(self, that: Self) -> bool:
        # [that] [self]
        return that._end_time <= self._start_time

    def is_same_camera(self, that: Self) -> bool:
        return self._camera == that._camera

    def overlap(self, that: Self) -> int:
        # time overlap between two tracks
        olen = 0
        if self._end_time <= that._start_time:
            # [self] [that]
            pass
        elif that._end_time <= self._start_time:
            # [that] [self]
            pass
        elif that._start_time <= self._start_time and self._end_time <= that._end_time:
            # [   that    ]
            #    [self]
            olen = (self._end_time - self._start_time).seconds
        elif self._start_time <= that._start_time and that._end_time <= self._end_time:
            #    [that]
            # [   self    ]
            olen = (that._end_time - that._start_time).seconds
        elif that._start_time < self._start_time and  that._end_time > self._start_time and that._end_time < self._end_time:
            # [that]
            #   [self]
            olen = (that._end_time - self._start_time).seconds
        elif self._start_time < that._start_time and self._end_time > that._start_time and self._end_time < that._end_time:
            #   [that]
            # [self]
            olen = (self._end_time - that._start_time).seconds
        else:
            # [that]
            # [self]
            olen = (self._end_time - self._start_time).seconds

        return olen

    def interval(self, that: Self) -> int:
        # time interval between two NOT overlapped tracks
        ilen = 86400
        if self._end_time <= that._start_time:
            # [self] [that]
            ilen = (that._start_time - self._end_time).seconds
        elif that._end_time <= self._start_time:
            # [that] [self]
            ilen = (that._end_time - self._start_time).seconds
        # elif that._start_time < self._start_time and self._end_time < that._end_time:
        #     # [   that    ]
        #     #    [self]
        #     pass
        # elif self._start_time < that._start_time and that._end_time < self._end_time:
        #     #    [that]
        #     # [   self    ]
        #     pass
        # elif that._start_time < self._start_time and that._end_time < self._end_time:
        #     # [that]
        #     #   [self]
        #     pass
        # elif self._start_time < that._start_time and self._end_time < that._end_time:
        #     #   [that]
        #     # [self]
        #     pass
        # else:
        #     # [that]
        #     # [self]
        #     pass
        return ilen

    def iop(self, that: Self) -> float:
        if self.is_before(that):
            # [self] [that]
            from_bbox = self._end_bbox
            to_bbox = that._start_bbox
        elif self.is_after(that):
            # [that] [self]
            from_bbox = that._end_bbox
            to_bbox = self._start_bbox
            pass
        else:
            raise ValueError(f"Invalid tracks: {self.name} and {that.name} have overlap")

        return iop_ratio(from_bbox, to_bbox)
# end


class DayTracks:

    def __init__(self):
        self._day: date = dt.date.day
        self._tracks: dict[str, Track] = {}
        self._dag = nx.DiGraph()
        self._log = logging.getLogger("DayTracks")

    def __getitem__(self, name):
        return self._tracks[name]

    @property
    def dag(self) -> nx.DiGraph:
        return self._dag

    def track(self, name) -> Track:
        return self._tracks[name]

    def add(self, track: Track):
        if self._dag.number_of_nodes() == 0:
            self._day = track.day

        assert self._day == track.day

        # skip the tracks composed by only "random_crop"
        # if track.random_crop_only:
        #     return

        self._day = track.day
        self._tracks[track.name] = track

    def analyze(self, min_duration: int, max_interval: int, random_crop_only: bool, tr: bool=False):
        self._add_nodes(min_duration, random_crop_only)
        self._add_edges(max_interval)

        if tr:
            self._dag = netx.transitive_reduction(self._dag, True, True)

    def _add_nodes(self, min_duration: int, random_crop_only: bool):
        for name in self._tracks:
            track = self._tracks[name]
            if not random_crop_only and track.is_random_crop_only:
                continue

            if track.duration < min_duration:
                continue

            self._dag.add_node(name)
        pass

    def _add_edges(self, max_interval: int):
        names: list[str] = cast(list[str], [n for n in self._dag])
        n = len(names)

        for i in range(n-1):
            ni = names[i]
            ti = self._tracks[ni]

            for j in range(i, n):
                nj = names[j]
                tj = self._tracks[nj]

                if ni == "1_13_DONE" and nj == "1_14_DONE":
                    interval = ti.interval(tj)

                # skip track with overlap (in seconds)
                if ti.overlap(tj) > 0:
                    continue

                # don't connect tracks with interval between them grater than
                #   max_interval (in seconds)
                interval = ti.interval(tj)
                if interval > max_interval:
                    continue

                if ti.is_before(tj):
                    self._dag.add_edge(ni, nj, weight=interval)
                elif ti.is_after(tj):
                    self._dag.add_edge(nj, ni, weight=interval)
        # end

    # def cleanup(self, tr: bool, from_same_cam: bool, to_same_cam: bool, min_iop: float):
    #     self.cleanup_transitive_reduction(tr)
    #     self.cleanup_by_same_cam(from_same_cam)
    #     self.cleanup_by_iop(min_iop)
    #     self._log.info(f"cleanup done")

    def cleanup_transitive_reduction(self, apply: bool=True):
        if not apply: return
        self._log.info(f"cleanup transitive_reduction")

        self._dag = netx.transitive_reduction(self._dag, True, True)

    def cleanup_by_degree(self, max_deg: int):
        self._cleanup_by_in_degree(max_deg)
        self._cleanup_by_out_degree(max_deg)

    def _cleanup_by_in_degree(self, max_deg: int):
        if max_deg <= 0: return
        self._log.info(f"cleanup _cleanup_by_in_degree")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.in_degree(n) <= max_deg:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            preds = list(G.predecessors(n))
            preds_interval = [
                (p, self._tracks[p].interval(n_track))
                for p in preds if self._tracks[p].camera == n_track.camera
            ]

            preds_interval = sort_by_key(preds_interval, key=lambda t: t[1])
            npi = len(preds_interval)

            for i in range(min(npi, max_deg)):
                to_keep.append(preds_interval[i][0])
            for i in range(max_deg, npi):
                to_remove.append(preds_interval[i][0])

            self._log.info(f"...    keep {to_keep} -> {n}")
            self._log.info(f"... removed {n} -> {to_remove}")
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for p in to_remove_all[n]:
                G.remove_edge(p, n)
    # end

    def _cleanup_by_out_degree(self, max_deg: int):
        if max_deg <= 0: return
        self._log.info(f"cleanup _cleanup_by_in_degree")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.in_degree(n) <= max_deg:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            succs = list(G.successors(n))
            succs_interval = [
                (s, n_track.interval(self._tracks[s]))
                for s in succs if self._tracks[s].camera == n_track.camera
            ]

            succs_interval = sort_by_key(succs_interval, key=lambda t: t[1])
            npi = len(succs_interval)

            for i in range(min(npi, max_deg)):
                to_keep.append(succs_interval[i][0])
            for i in range(max_deg, npi):
                to_remove.append(succs_interval[i][0])

            self._log.info(f"...    keep {to_keep} -> {n}")
            self._log.info(f"... removed {n} -> {to_remove}")
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for s in to_remove_all[n]:
                G.remove_edge(n, s)
    # end

    # -----------------------------------------------------------------------

    def cleanup_by_same_cam(self, apply: bool=True):
        self._cleanup_from_same_cam(apply)
        self._cleanup_to_same_cam(apply)

    def _cleanup_from_same_cam(self, apply:bool=True):
        if not apply: return
        self._log.info(f"cleanup from_same_cam")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.in_degree(n) < 2:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            preds = list(G.predecessors(n))
            for p in preds:
                p_track: Track = self._tracks[p]
                if n_track.is_same_camera(p_track):
                    to_keep.append(p)
                else:
                    to_remove.append(p)
            # end

            # keep_interval   = [self._tracks[p].interval(self._tracks[n]) for p in to_keep]
            # remove_interval = [self._tracks[p].interval(self._tracks[n]) for p in to_remove]
            keep_interval = [G.edges[(p, n)]["weight"] for p in to_keep]
            remove_interval = [G.edges[(p, n)]["weight"] for p in to_remove]

            if len(to_remove) == 0:
                continue

            if len(to_keep) == 0:
                self._log.warning(f"... NOT remove {to_remove} -> {n}/{remove_interval}")
                continue

            self._log.info(f"...    keep {to_keep} -> {n}/{keep_interval}")
            self._log.info(f"... removed {to_remove} -> {n}/{remove_interval}")
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for p in to_remove_all[n]:
                G.remove_edge(p, n)
        pass

    def _cleanup_to_same_cam(self, apply: bool=True):
        if not apply: return
        self._log.info(f"cleanup to_same_cam")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.out_degree(n) < 2:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            succs = list(G.successors(n))
            for s in succs:
                s_track: Track = self._tracks[s]
                if n_track.is_same_camera(s_track):
                    to_keep.append(s)
                else:
                    to_remove.append(s)
            # end

            # keep_interval   = [self._tracks[n].interval(self._tracks[s]) for s in to_keep]
            # remove_interval = [self._tracks[n].interval(self._tracks[s]) for s in to_remove]
            keep_interval = [G.edges[(n, s)]["weight"] for s in to_keep]
            remove_interval = [G.edges[(n, s)]["weight"] for s in to_remove]

            if len(to_remove) == 0:
                continue

            if len(to_keep) == 0:
                self._log.warning(f"... NOT remove {n} -> {to_remove}/{remove_interval}")
                continue

            self._log.info(f"...    keep {n} -> {to_keep}/{keep_interval}")
            self._log.info(f"... removed {n} -> {to_remove}/{remove_interval}")
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for s in to_remove_all[n]:
                G.remove_edge(n, s)
        pass

    # -----------------------------------------------------------------------

    def cleanup_by_iop(self, apply: bool=True):
        self._cleanup_from_same_cam(apply)
        self._cleanup_to_same_cam(apply)

    def _cleanup_out_iop(self, min_iop: float=0.5):
        if min_iop is None or min_iop <= 0: return
        self._log.info(f"cleanup out-iop")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.out_degree(n) < 2:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            succs = list(G.successors(n))

            iop_list = [
                (n,s, n_track.iop(self._tracks[s]))
                for s in succs
                if self._tracks[s].is_same_camera(n_track)
            ]

            for n, s, iop in iop_list:
                if iop <= min_iop:
                    to_remove.append(s)
                else:
                    to_keep.append(s)
            # end
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for s in to_remove_all[n]:
                G.remove_edge(n, s)
        pass

    def _cleanup_in_iop(self, min_iop: float):
        if min_iop is None or min_iop <= 0: return
        self._log.info(f"cleanup in-iop")

        G = self._dag
        to_remove_all: dict[str, list[str]] = {}
        nodes = [n for n in G]
        for n in nodes:
            if G.in_degree(n) < 2:
                continue

            to_keep = []
            to_remove = []
            n_track: Track = self._tracks[n]
            precs = list(G.predecessors(n))

            iop_list = [
                (p, n, self._tracks[p].iop(n_track))
                for p in precs
                if self._tracks[p].is_same_camera(n_track)
            ]

            for p, n, iop in iop_list:
                if iop <= min_iop:
                    to_remove.append(p)
                else:
                    to_keep.append(p)
            # end
            to_remove_all[n] = to_remove
        # end

        for n in to_remove_all:
            for p in to_remove_all[n]:
                G.remove_edge(p, n)
        pass

    # -----------------------------------------------------------------------

    # def reconnect_by_interval(self, max_interval: int):
    #     # analyze tracks by camera
    #     self._log.info(f"reconnect_by_interval")
    #
    #     G = self._dag
    #     clusters = netx.transitive_clusters(G)
    #
    #     cluster_tracks = {
    #         n: ClusterTrack(self, n, clusters[n])
    #         for n in clusters
    #     }
    #
    #     pass

    # -----------------------------------------------------------------------

    # def reconnect_by_embedding(self, max_interval: int, threshold: float):
    #     # analyze tracks by camera
    #     self._log.info(f"reconnect_by_embedding")
    #
    #     G = self._dag
    #     clusters = netx.transitive_clusters(G)
    #     clusters = netx.remove_duplicate_clusters(clusters)
    #
    #     cluster_tracks = {
    #         n: ClusterTrack(self, n, clusters[n])
    #         for n in clusters
    #     }
    #     pass

    # -----------------------------------------------------------------------

    def track_statistics(self) -> dict[str, list[int]]:
        G = self._dag

        durations = [
            self._tracks[n].duration
            for n in G.nodes
        ]

        intervals = [
            self._tracks[e[0]].interval(self._tracks[e[1]])
            for e in G.edges
        ]

        # intervals = []
        # # compute min, max, mean intervals between two tracks
        # names = list(self._tracks.keys())
        # n = len(names)
        #
        # nl = names[-1]
        # tl = self._tracks[nl]
        # durations.append(tl.duration)
        #
        # for i in range(n - 1):
        #     ni = names[i]
        #     ti = self._tracks[ni]
        #     durations.append(ti.duration)
        #
        #     for j in range(i, n):
        #         nj = names[j]
        #         tj = self._tracks[nj]
        #
        #         # skip track with overlap (in seconds)
        #         if ti.overlap(tj) > 0:
        #             continue
        #
        #         # don't connect tracks with interval between them grater than
        #         #   max_interval (in seconds)
        #         interval = ti.interval(tj)
        #
        #         intervals.append(interval)
        #     # end for j
        # # end for i
        # if len(intervals) == 0:
        #     self._log.debug("   no tracks found")
        #     return {}

        # intervals = np.array(intervals)
        # self._log.print("   Interval statistics:")
        # self._log.print("      min:", intervals.min())
        # self._log.print("      max:", intervals.max())
        # self._log.print("      mean:", intervals.mean())
        # self._log.print("      stdv:", intervals.std())

        return {
            "durations": durations,
            "intervals":  intervals
        }
    # end

    def save_dag(self, save_file: str|Path|None=None):
        if isinstance(save_file, str):
            save_file = Path(save_file)

        # sday = self._day.strftime("%Y-%m-%d")
        save_dir = save_file.parent
        # stem = save_file.stem
        # ext = save_file.suffix

        save_dir.mkdir(parents=True, exist_ok=True)
        # save_file = save_dir / f"{stem}-{sday}{ext}"

        if save_file.name.endswith(".gml"):
            nx.write_gml(self._dag, str(save_file))
        else:
            raise ValueError(f"Unsupported file format {save_file.name}")
        pass
# end


CLUSTER_TRACK_ID = 1

class ClusterTrack:
    def __init__(self, dts: DayTracks, node:str, cluster: set[str]):
        global CLUSTER_TRACK_ID

        assert is_instance(dts, DayTracks)
        assert is_instance(cluster, set[str])

        self._ctid = CLUSTER_TRACK_ID
        CLUSTER_TRACK_ID += 1

        self._dts: DayTracks = dts
        self._start_node: str = node
        self._cluster: set[str] = cluster

        self._sources = None
        self._sinks = None
        self._start_time = None
        self._end_time = None
        self._analyze()

    def _analyze(self):
        G = self._dts.dag
        self._sources = [n
            for n in self._cluster
            if G.in_degree(n) == 0
        ]

        self._sinks = [n
            for n in self._cluster
                 if G.out_degree(n) == 0
        ]

    @property
    def ctid(self):
        return self._ctid

    @property
    def is_path(self):
        return len(self._sources) == 1 and len(self._sinks) == 1

    @property
    def start_time(self) -> date:
        if self._start_time is None:
            sources = self._sources
            start_time = self._dts.track(sources[0]).start_time
            for n in sources:
                n_start_time = self._dts.track(n).start_time
                if n_start_time < start_time:
                    start_time = n_start_time
            self._start_time = start_time
        return self._start_time

    @property
    def end_time(self)-> date:
        if self._end_time is None:
            sinks = self._sinks
            end_time = self._dts.track(sinks[0]).end_time
            for n in sinks:
                n_end_time = self._dts.track(n).end_time
                if n_end_time > end_time:
                    end_time = n_end_time
            self._end_time = end_time
        return self._end_time

    def is_before(self, that: Self) -> bool:
        return self.end_time <= that.start_time

    def is_after(self, that: Self) -> bool:
        return that.end_time <= self.start_time

    def interval(self, that: Self) -> float:
        ilen = 86400
        if self.end_time < that.start_time:
            # [self] [that]
            ilen = (that.start_time - self.end_time).seconds
        elif that.end_time < self.start_time:
            # [that] [self]
            ilen = (self.start_time - that.end_time).seconds
        else:
            pass
        return ilen
# end


# ---------------------------------------------------------------------------
# TimeTracksAnalyzer
# ---------------------------------------------------------------------------

class DAGTracksAnalyzer:

    def __init__(self, CONFIG: JSONConfiguration):
        self.CONFIG = CONFIG

        # min track duration (in seconds)
        self.min_duration:int = CONFIG.get("time_tracks.min_duration", 1)
        # max interval between two tracks to consider them 'connected'(in seconds)
        self.max_interval: int = CONFIG.get("time_tracks.max_interval", 300)
        # is to accept tracks with 'more_persons=true'
        self.more_persons: bool = CONFIG.get("time_tracks.more_persons", True)
        # folder MUST be present
        self.validate_folders: list[str] = CONFIG.get("time_tracks.validate_folders", ["random_crop"])
        # if to apply the DAG transitive reduction to the inferred DAG
        self.transitive_reduction: bool = CONFIG.get("time_tracks.transitive_reduction", True)
        # if to save the DAG in the transk director
        self.save = CONFIG.get("time_tracks.save", False)

        # [CM} MA A CHE SERVE????
        self.random_crop_only: bool = CONFIG.get("time_tracks.random_crop_only", True)

        self._tracks_root = None
        self._day_tracks_map:dict[date, DayTracks] =  defaultdict(lambda: DayTracks())

        self._log = logging.getLogger("TimeTracksAnalyzer")
    # end

    def analyze(
        self,
        tracks_root: str|Path,
        dir_pattern: str="",
        max_tracks: int = 0,
    ):
        def _split(name) -> tuple[int, int]:
            parts = name.split("_")
            return int(parts[0]), int(parts[1])

        # track_root: root of the directory to analyze
        # dir_pattern: string must be contained in the track directory
        # max_tracks:[DEBUGGING ONLY] limit the number of tracks to process

        if dir_pattern is None: dir_pattern = ""
        if max_tracks <= 0: max_tracks = 2147483647
        if isinstance(tracks_root, str): tracks_root = Path(tracks_root)

        self._tracks_root = tracks_root

        # min_duration: minimum time length of the track (in seconds)
        # max_interval: maximum time length between two tracks (in seconds)
        # tr: if to apply the transitive reduction
        self._log.info(f"Analyzing {tracks_root}/{dir_pattern}")

        min_duration = self.min_duration
        max_interval = self.max_interval
        tr = self.transitive_reduction
        random_crop_only = self.random_crop_only

        # scan the directory and collect all tracks
        # organize them by date
        # for track in self._tracks_root.iterdir():
        #     if track.name.endswith("_DONE"):
        #         self._analyze_track(track)
        #
        #     g += 1
        #     if g >= max_tracks:
        #         break
        # end

        track_names = [
            track_dir.name
            for track_dir in tracks_root.iterdir()
            if track_dir.name.endswith("_DONE") and (dir_pattern in track_dir.name)
        ]
        track_names = sort_by_key(track_names, key=_split)
        n_tracks = len(track_names)

        for i, track_name in enumerate(track_names):
            track_dir = tracks_root / track_name

            if not self._is_valid_track(track_dir):
                continue

            self._log.infot(f"... {track_name} ({i+1}/{n_tracks})")

            self._analyze_track(track_dir)

            if i >= max_tracks:
                break
        # end

        # analyze all track on day-basis
        for day in self._day_tracks_map:
            self._day_tracks_map[day].analyze(min_duration, max_interval, random_crop_only, tr)

        self.save_dags()

        self._log.info(f"Done")
    # end

    def _is_valid_track(self, track_dir: Path):
        if not track_dir.name.endswith("_DONE"):
            return False

        # check if all folders in 'valid_folders' are present
        for valid_name in self.validate_folders:
            if not (track_dir / valid_name).exists():
                return False

        # check if 'more-Persons' is accepted
        if self.more_persons:
            return True

        # ensure the track has 'more_persons=false'
        meta = jsonx.load(track_dir / "meta.json")
        more_persons = meta["more_persons"]
        return not more_persons

    def _analyze_track(self, track_dir: Path):

        meta_path = track_dir / "meta.json"
        if not meta_path.exists():
            return

        meta = jsonx.load(meta_path)
        track = Track(track_dir, meta)

        self._day_tracks_map[track.day].add(track)
        pass

    # def cleanup(self, tr: bool=True, from_same_cam: bool=True, to_same_cam: bool=True, min_iop: float = 0.1):
    #     # tr: transitive reduction
    #     # from_same_cam: prefer connections from the same camera
    #     #   to_same_cam: prefer connections   to the same camera
    #     # min_iop: remove connections if the iop of the bounding boxes is less than the specific value
    #     self._log.info("cleanup")
    #     for day in self._day_tracks_map:
    #         self._log.info(f"   {day}")
    #         self._day_tracks_map[day].cleanup(tr, from_same_cam, to_same_cam, min_iop)

    def cleanup_transitive_reduction(self, tr):
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            self._day_tracks_map[day].cleanup_transitive_reduction(tr)

    def cleanup_by_degree(self, max_degree: int=5):
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            self._day_tracks_map[day].cleanup_by_degree(max_degree)

    def cleanup_by_same_cam(self, to_same_cam):
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            self._day_tracks_map[day].cleanup_by_same_cam(to_same_cam)

    def cleanup_by_iop(self, min_iop):
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            self._day_tracks_map[day].cleanup_by_iop(min_iop)

    def reconnect_by_interval(self, max_interval: int=0):
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            self._day_tracks_map[day].reconnect_by_interval(max_interval)

    # def reconnect_by_embedding(self, max_interval:int=0, threshold: float=0.92):
    #     for day in self._day_tracks_map:
    #         self._log.info(f"   {day}")
    #         self._day_tracks_map[day].reconnect_by_embedding(max_interval, threshold)

    def track_statistics(self) -> dict[date, dict]:
        self._log.info("interval_statistics")
        stats: dict[date, dict] = {}
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            day_stats = self._day_tracks_map[day].track_statistics()
            stats[day] = day_stats
        return stats

    def save_dags(self, save_dir: str|Path|None=None):
        if save_dir is None: save_dir = self._tracks_root
        if isinstance(save_dir, str): save_dir = Path(save_dir)

        self._log.info(f"Save DAG in {save_dir}")
        for day in self._day_tracks_map:
            self._log.info(f"   {day}")
            sday = day.strftime("%Y-%m-%d")
            self._day_tracks_map[day].save_dag(save_dir / f"dag_tracks_{sday}.gml")
# end



# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
