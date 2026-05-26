from pathlib import Path
from stdlib.sortedx import sort_by_comparator, lexicographic_cmp
from stdlib import jsonx
from datetime import datetime


def list_done_dirs(root_path: Path, cam_id: int):
    done_dirs = []
    cam_prefix = f"{cam_id}_"
    for done_dir in root_path.iterdir():
        if not done_dir.name.endswith("_DONE"):
            continue
        if not done_dir.name.startswith(cam_prefix):
            continue
        done_dirs.append(done_dir.name)
    return sort_by_comparator(done_dirs, lexicographic_cmp)




def main():
    root_path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")

    for cam_id in [0,1,2,3]:
        done_dirs = list_done_dirs(root_path, cam_id)
        n_dirs = len(done_dirs)
        for i in range(n_dirs-1):
            j = i+1
            i_done = done_dirs[i]
            i_meta = jsonx.load(root_path / i_done / "meta.json")
            present_end = datetime.strptime(i_meta["present_end"], "%Y-%m-%d %H:%M:%S")
            j_done = done_dirs[j]
            j_meta = jsonx.load(root_path / j_done / "meta.json")
            present_start = datetime.strptime(j_meta["present_start"], "%Y-%m-%d %H:%M:%S")

            total_second = (present_start - present_end).total_seconds()
            print(f"{i_done}->{j_done}: {total_second}")
            pass
        pass
    pass



if __name__ == "__main__":
    main()