from pathlib import Path
ROOT_DIR = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result_3\new\20260506_ 91302")

def main():

    for cam_track in ROOT_DIR.glob("*_DONE"):
        print(cam_track)





if __name__ == "__main__":
    main()
