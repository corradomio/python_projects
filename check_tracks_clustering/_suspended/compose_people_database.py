from collections import defaultdict
from pathlib import Path
import stdlib.loggingx as logging
from stdlib import jsonx
from stdlib import csvx

from people_database import PeopleDatabase

TRACKS_ROOT: Path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260513_155653")
PEOPLE_DATABASE_ROOT: Path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\.people_database")


def has_person_id(track_dir: Path) -> bool:
    for file in track_dir.iterdir():
        if file.name.startswith("whoami_"):
            return True
    return False


def has_face(track_dir: Path) -> bool:
    face_dir = track_dir / "face"
    return face_dir.exists()


def save_person_id(track_dir: Path, person_id: int, similarity: float):
    whoami = {
        "person_id": person_id,
        "similarity": similarity
    }
    whoami_path = track_dir / f"whoami_{person_id}.json"
    jsonx.dump(whoami, whoami_path)

    whoami_path = track_dir / f"random_crop/whoami_{person_id}.json"
    jsonx.dump(whoami, whoami_path)
    pass


def load_person_id(track_dir: Path) -> tuple[int, float]:
    selected_file = None
    for file in track_dir.iterdir():
        if file.name.startswith("whoami_"):
            selected_file = file
            break

    if selected_file is None:
        return -1, 0.

    # "person_id": <pid:int>
    # "similariry": <sim:float>
    data = jsonx.load(selected_file)
    return data["person_id"], data["similarity"]


# ---------------------------------------------------------------------------

def main():
    log = logging.getLogger("main")

    people_dict = defaultdict(lambda : list())

    pdb = PeopleDatabase(root=PEOPLE_DATABASE_ROOT, top_k=-1, threshold=0.92)

    log.info(f"Processing {TRACKS_ROOT}")
    for track_dir, _, _ in TRACKS_ROOT.walk():
        if not track_dir.name.endswith("_DONE"):
            continue

        if not has_face(track_dir):
            continue

        if has_person_id(track_dir):
            person_id, similarity = load_person_id(track_dir)
            people_dict[person_id].append(track_dir.name)
            continue

        log.infot(f"... {track_dir.parent.parent.name}/{track_dir.parent.name}/{track_dir.name}")

        random_crop = track_dir / "random_crop"
        face_recognition = track_dir / "face_recognition"
        face_recognition_fallback = track_dir / "face_recognition_fallback"

        if not random_crop.exists():
            continue

        images_dirs = [random_crop, face_recognition, face_recognition_fallback]

        person_id, similarity, valid = pdb.find_people_id(images_dirs)
        if person_id == -1 or not valid:
            pdb.create_person_id(images_dirs)
        else:
            pdb.update_person_id(person_id, images_dirs)

        save_person_id(track_dir, person_id, similarity)

        people_dict[person_id].append(track_dir.name)
    pass

    people_list = [
        [f"P{person_id}", ",".join(people_dict[person_id])]
        for person_id in people_dict
    ]

    csvx.dump(people_list, f"labels_{TRACKS_ROOT.name}.csv", header=["Person","Tracks"])
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    main()
