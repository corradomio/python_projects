from lmdbenv import LmdbEnv


def main():
    with LmdbEnv.open("test.lmdb", max_dbs=16) as env:
        kv = env.select("kv", str, int)
        kv['one'] = 1
        kv['two'] = 2

        if "three" in kv:
            print("ok")

        print(kv['one'])
        print(kv['two'])
    pass


if __name__ == "__main__":
    main()
