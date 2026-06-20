from datetime import date
from stdlib import loggingx as logging
import matplotlib.pyplot as plt

from stdlib.jsonx import JSONConfiguration

TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result"
# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-03-11"
# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-03-26"

# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_saved_2"
# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_saved_3"
# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_saved_4"
# TRACKS_ROOT = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data"

from dag_tracks_analyzer import DAGTracksAnalyzer

INT_INF = 6535

def print_t_stats(day_stats: dict[date, dict]):
    for day in day_stats:
        sday = day.strftime("%Y-%m-%d")
        # {
        #   "durations": durations,
        #   "intervals":  intervals
        # }
        stats = day_stats[day]
        durations = stats["durations"]
        durations = [t for t in durations if t < 60]
        intervals = stats["intervals"]
        intervals = [t for t in intervals if t < 60]

        plt.clf()
        plt.hist(intervals, bins=50)
        plt.title("intervals")
        plt.xlabel("seconds")
        plt.savefig("intervals.png", dpi=300)

        plt.clf()
        plt.hist(durations, bins=20)
        plt.title("duration")
        plt.xlabel("seconds")
        plt.savefig("durations.png", dpi=300)



def main():
    CONFIG = JSONConfiguration.load("config_post_dev.json")
    ta = DAGTracksAnalyzer(CONFIG)

    # t_stats = ta.track_statistics()
    # print_t_stats(t_stats)

    ta.analyze(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-08\20260513_155653")

    # ta.save("./dags/1_tracks-full.gml")
    ta.cleanup_by_degree(4)
    # ta.save("./dags/2_tracks-degree.gml")
    ta.cleanup_by_iop(0.5)
    # ta.save("./dags/3_tracks-iop.gml")
    ta.cleanup_by_same_cam(True)
    # ta.save("./dags/4_tracks-cam.gml")
    ta.cleanup_transitive_reduction(True)

    ta.save_dags("./dags/5_tracks-tr.gml")

    # ta.reconnect_tracks(max_interval=5*60)
    t_stats = ta.track_statistics()
    print_t_stats(t_stats)
    pass



if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    log = logging.getLogger("main")
    log.info("Logging initialized")
    main()

