from pathlib import Path

ROOT = Path(r"D:\Projects.ebtic.datasets\.faces_gallery")

def main():

    for i in range(60):
        person_folder = ROOT / f"Person-{i+1}"
        person_folder.mkdir(exist_ok=True)
    pass


if __name__ == "__main__":
    main()
