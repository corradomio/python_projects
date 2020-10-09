import os

ROOT_PATH = "Damiani"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")
held_out_directory = os.path.join(ROOT_PATH, "HeldOut")


def sel_held_out(data_directory):
    for d in os.listdir(data_directory):
        path = os.path.join(data_directory, d)
        if not os.path.isdir(path):
            continue
        for f in os.listdir(path):
            ppm = os.path.join(path, f)
            if not ppm.endswith("_00000.ppm"):
                continue

            # create the held out directory
            hod = os.path.join(held_out_directory, d)
            if not os.path.exists(hod):
                os.mkdir(hod)

            hof = os.path.join(hod, f)
            # move te file
            if not os.path.exists(hof):
                os.rename(ppm, hof)
            break
    pass


def main():
    # sel_held_out(train_data_directory)
    sel_held_out(test_data_directory)


if __name__ == "__main__":
    main()
