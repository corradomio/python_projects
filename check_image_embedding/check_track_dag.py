import datetime
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from stdlib import jsonx
from stdlib import loggingx as logging
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
import networkx as nx


LEN_DONE = len("_DONE")

class TrackDag:
    #
    # It creates a graph between the tracks based on the timestamps
    #
    def __init__(self, CONFIG: JSONConfiguration):
        self.CONFIG = CONFIG

        self.gap_max = CONFIG.get("track_merger.params.gap_max", 30)


    def analyze_tracks(self, root_path: Path):
        meta_dict: dict[str, dict] = self._load_meta_dict(root_path)

        tracks = list(meta_dict.keys())
        n_tracks = len(tracks)

        track_dag = nx.DiGraph()
        track_dag.add_nodes_from(tracks)

        for i in range(n_tracks-1):
            i_track = tracks[i]
            i_meta = meta_dict[i_track]
            present_end = datetime.strptime(i_meta["present_end"], "%Y-%m-%d %H:%M:%S")
            for j in range(i+1, n_tracks):
                j_track = tracks[j]
                j_meta = meta_dict[j_track]
                present_start = datetime.strptime(j_meta["present_start"], "%Y-%m-%d %H:%M:%S")

                total_second = (present_start - present_end).total_seconds()
                if total_second < 0:
                    continue
                if total_second > self.gap_max:
                    continue

                track_dag.add_edge(i_track, j_track)
                print(f"{i_track} -> {j_track}")
            pass

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        nx.draw(track_dag)
        plt.savefig(f"track_dag-{self.gap_max}.png", dpi=300)
        nx.write_graphml(track_dag, f"track_dag-{self.gap_max}.graphml")
        pass


    def _load_meta_dict(self, root_path: Path):
        meta_dict = {}
        for done_dir in root_path.iterdir():
            if not done_dir.name.endswith("_DONE"):
                continue

            meta = jsonx.load(done_dir / "meta.json")
            done_name = done_dir.name[:-LEN_DONE]
            meta_dict[done_name] = meta
        return meta_dict





def main():
    CONFIG = JSONConfiguration.load(r"D:\Projects.ebtic\project.diwang\lab_monitoring\config_post_dev.json")

    root_path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")

    track_dag = TrackDag(CONFIG)
    track_dag.analyze_tracks(root_path)
    pass


if __name__ == "__main__":
    main()
