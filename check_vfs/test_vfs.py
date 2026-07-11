import vfs
#
# Local file system
#

def test_lfs():
    lfs = vfs.create("vfs/local")
    assert not lfs.connected
    lfs.connect()
    assert lfs.connected
    lfs.disconnect()
    assert not lfs.connected

def test_lfs_file():
    lfs = vfs.create(".store/local")
    with lfs.connect():

        root = lfs.root
        assert root.exists()
        assert root.is_dir()
        assert not root.is_file()
        assert root.is_local()

        not_existent = root / "not_existent"
        assert not not_existent.exists()

        afile = root / "dir/afile.txt"
        assert not afile.exists()
        afile.mkfile(recursive=True, exist_ok=True)
        assert afile.exists()
        assert afile.is_file()
        assert afile.parent.exists()
        assert afile.parent.is_dir()

        afile.delete()
        assert not afile.exists()
        assert not afile.is_file()
        assert not afile.is_dir()
        assert afile.parent.exists()
        assert afile.parent.is_dir()
    # end


def test_lfs_copy():
    lfs = vfs.create(".store")
    with lfs.connect():
        file = lfs.root / "afile.txt"
        assert file.is_file()

        dir = lfs.root / "local/dir"
        assert dir.exists()

        # file -> dir
        file.copy(dir)
        clone = dir / file.name
        assert clone.is_file()

        # file -> file (not exist)
        clone = dir / "new.txt"
        assert not clone.exists()
        file.copy(clone)
        assert clone.is_file()

        # file -> file (exist)
        file.copy(clone)
        assert clone.is_file()

        clone.delete()
        assert not clone.exists()
    # end
# end


def test_ftsfs():
    rfs = vfs.create("ftp://localhost/ftp", username="jianyi", password="lin")
    assert not rfs.connected
    rfs.connect()
    assert rfs.connected
    rfs.disconnect()
    assert not rfs.connected

def test_sftsfs():
    rfs = vfs.create("sftp://localhost/ftp", username="jianyi", password="lin")
    assert not rfs.connected
    rfs.connect()
    assert rfs.connected
    rfs.disconnect()
    assert not rfs.connected

