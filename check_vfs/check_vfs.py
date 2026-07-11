import vfs


def check_vfs(rfs):
    with rfs.connect():
        assert rfs.connected
        root = rfs.root
        print(root)
        for d in root.iterdir():
            print(d, d.is_dir(), d.is_file())
            child = d / "child"
            parent = child.parent

            assert d == parent

        (root / "a/b/c").mkdir(exist_ok=True)
        (root / "a/b/c/f.txt").mkfile(exist_ok=True)

    assert not rfs.connected


def main():

    check_vfs(vfs.create("file://jianyi/lfs"))
    check_vfs(vfs.create("ftp://localhost/ftp", username="jianyi", password="lin"))
    check_vfs(vfs.create("sftp://localhost/sftp", username="jianyi", password="lin"))

    pass



if __name__ == "__main__":
    main()
