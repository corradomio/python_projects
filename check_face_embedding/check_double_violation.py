import contextlib
from pathlib import Path
from stdlib import jsonx


VIOLATIONS =[
    "not_dress_well", "not_glove_well",
    "unauthorised_access", "unauthorised_machine_touching_B", "unauthorised_operation_A"
]


def main():
    with open('violations.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            ROOT = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data")
            for meta_file in ROOT.rglob("meta.json"):
                data = jsonx.load(meta_file)
                v_count = 0
                for v in VIOLATIONS:
                    if v in data:
                        v_count += 1
                if v_count > 1:
                    print(meta_file, ":", v_count)

    pass


if __name__ == "__main__":
    main()
