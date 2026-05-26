from pathlib import Path
ROOT_DIR = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result_3\new\20260506_ 91302")

def main():

    total = 0
    dress = 0
    for cam_track in ROOT_DIR.glob("*_DONE"):
        total += 1
        dress_well = cam_track / "dress_well"
        not_dress_well = cam_track / "not_dress_well"

        if dress_well.exists() or not_dress_well.exists():
            continue

        dress += 1
        print(cam_track)

    print("total:", total, "dress:", dress)





if __name__ == "__main__":
    main()
