import shutil
from pathlib import Path
from typing import Generator, cast

from ..utils import _split_url, parent_of, join_with
from ..vfs import VFileSystem, VPath, VStream, VFilesystemException


# ---------------------------------------------------------------------------
# LocalPath
# ---------------------------------------------------------------------------

class LocalPath(VPath):
    def __init__(self, fs: LocalFileSystem, path: str):
        super().__init__(path)
        self.fs = fs
        self._path: Path = Path(self.path)

    def exists(self) -> bool:
        return self._path.exists()

    def is_file(self) -> bool:
        return self._path.is_file()

    def is_dir(self) -> bool:
        return self._path.is_dir()

    def iterdir(self) -> Generator['LocalPath', None, None]:
        if self._path.exists():
            for f in self._path.iterdir():
                yield LocalPath(self.fs, str(f))

    @property
    def parent(self) -> 'LocalPath':
        if self.path == self.fs.root.path:
            return self
        parent_path = parent_of(self.path)
        return LocalPath(self.fs, parent_path)

    def __truediv__(self, child: str) -> 'LocalPath':
        assert isinstance(child, str)
        assert child != ".." and child != "."
        child_path = join_with(self.path, child)
        return LocalPath(self.fs, child_path)

    def as_path(self) ->Path:
        return self._path

    def mkdir(self, recursive: bool=False, exist_ok=False):
        self._path.mkdir(parents=recursive, exist_ok=exist_ok)

    def mkfile(self, recursive: bool=False, exist_ok=False):
        if recursive:
            self.parent.mkdir(recursive=True, exist_ok=True)
        self._path.touch(exist_ok=exist_ok)

    def copy(self, dest: VPath, recursive: bool = False):
        if not dest.is_local():
            super().copy(dest, recursive)

        # shutil.copyfileobj(fsrc, fdst[, length])
        # shutil.copyfile(src, dst, *, follow_symlinks=True)
        # shutil.copy(src, dst, *, follow_symlinks=True)
        # shutil.copy2(src, dst, *, follow_symlinks=True)
        # shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
        # shutil.rmtree(path, ignore_errors=False, onerror=None, *, onexc=None, dir_fd=None)
        # shutil.move(src, dst, copy_function=copy2).

        src_path = self._path
        dst_path = cast(LocalPath, dest)._path
        if src_path.is_file() and (dst_path.is_file() or not dst_path.exists()):
            if recursive:
                dest.parent.mkdir(recursive=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
        elif src_path.is_file() and dst_path.is_dir():
            shutil.copy2(src_path, dst_path)
        elif src_path.is_dir() and dst_path.is_dir():
            shutil.copytree(src_path, dst_path)
        elif src_path.is_dir() and not dst_path.exists() and recursive:
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path)
        else:
            raise VFilesystemException(f"Cannot copy {src_path} to {dst_path}")
        pass

    def _copy(self, dest: VPath):
        assert isinstance(dest, LocalPath)
        src_path = self.as_path()
        dst_path = dest.as_path()
        shutil.copy2(src_path, dst_path)

    def _delete(self):
        self._path.unlink(missing_ok=True)

    def open(self, mode: str, **kwargs):
        return open(self.path, mode=mode, **kwargs)
# end


# ---------------------------------------------------------------------------
# LocalFileSystem
# ---------------------------------------------------------------------------

class LocalFileSystem(VFileSystem):
    def __init__(self, url: str|Path, **kwargs):
        super().__init__()
        assert isinstance(url, (str, Path))
        self._url: str = str(url)
        self._root: Path = None

    def connect(self):
        assert self._root is None

        host, path = _split_url(self._url)

        self._root = LocalPath(self, path)
        return self

    def disconnect(self):
        self._root = None
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
