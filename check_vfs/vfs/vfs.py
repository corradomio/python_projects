from pathlib import Path
from typing import Self, Generator, Literal

from .utils import name_of, stem_of, suffix_of, normalize


OPEN_MODES = Literal[
    'r',  'w',   'a',
    'r+', 'w+',  'a+',
    'rb', 'wb',  'ab',
    'rb+','wb+', 'ab+'
]


class VFilesystemException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
    pass


class VStream:
    def __init__(self, path: VPath):
        self.path: VPath = path


class VPath:
    def __init__(self, path: str):
        assert isinstance(path, str)
        self.path = normalize(path)

    def is_local(self):
        return self.__class__.__name__ == 'LocalPath'

    def is_remote(self):
        return self.__class__.__name__ != 'LocalPath'

    def exists(self) -> bool:
        ...

    def is_file(self) -> bool:
        ...

    def is_dir(self) -> bool:
        ...

    def iterdir(self) -> Generator[VPath, None, None]:
        ...

    @property
    def parent(self) -> VPath:
        return self

    @property
    def name(self) -> str:
        return name_of(self.path)

    @property
    def stem(self) -> str:
        return stem_of(self.path)

    @property
    def suffix(self) -> str:
        return suffix_of(self.path)

    # this / child
    def __truediv__(self, child: str) -> Self:
        ...

    # this == that
    def __eq__(self, other: VPath) -> bool:
        return self.path == other.path

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.path}]"

    def as_path(self) ->Path:
        return Path(self.path)

    def mkdir(self, recursive=False, exist_ok=False):
        if self.exists() and not exist_ok:
            raise FileExistsError(f"{self.path} already exists")
        if self.exists(): return
        if recursive:
            self.parent.mkdir(exist_ok=True)
        self._mkdir()
        return self

    def _mkdir(self):
        # low level 'mkdir'
        ...

    def mkfile(self, recursive=False, exist_ok=False):
        if self.exists() and not exist_ok:
            raise FileExistsError(f"{self.path} already exists")
        if self.exists(): return
        if recursive:
            self.parent.mkdir(exist_ok=True)
        self._mkfile()

    def _mkfile(self):
        # low level 'mkfile''
        ...

    def copy(self, dest: VPath, recursive: bool = False):
        # file -> file|not existent
        # file -> directory (same name)
        # directory -> directory (recursive)
        #
        # Some transfers can be more efficient.
        # For example:
        #   local  -> remote (upload)
        #   remote -> local (download)
        #
        #
        if self.is_file() and dest.is_file():
            self._copyfile(dest)

        elif self.is_file() and dest.is_dir():
            self._copyfile(dest / self.name)
        elif self.is_dir() and dest.is_dir():
            # copy files
            for sfile in self.iterdir():
                if sfile.is_dir(): continue
                dfile = dest / sfile.name
                self._copyfile(dfile)
            pass

            # DONT' copy directories if not recursive
            if not recursive:
                return

            # copy directories
            for sdir in self.iterdir():
                if sdir.is_dir(): continue
                ddir = dest / sdir.name
                ddir.mkdir(recursive=True, exist_ok=True)
                sdir.copy(ddir, recursive=True)
            # end
        else:
            raise NotADirectoryError(f"{dest} is not a directory")
        ...

    def _copyfile(self, dest: VPath):
        if self.is_local() and dest.is_local():
            self._copy(dest)
        elif self.is_local() and dest.is_remote():
            dest._copyfrom(self.as_path())
        elif self.is_remote() and dest.is_local():
            dest._copyto(self.as_path())


    def _copy(self, dest: VPath):
        # low level copy: file -> file
        #   local -> local
        #   remote -> remote
        ...

    def _copyto(self, dest: Path):
        # low level copy: file (remote) -> file (local)
        ...

    def _copyfrom(self, src: Path):
        # low level copy: file (remote) <- file (local)
        ...

    # r w a r+ w+ a+ b
    def open(self, mode: OPEN_MODES, **kwargs) -> VStream:
        ...

    def delete(self, recursive=False):
        # file
        # directory (recursive)
        if not self.exists():
            return
        elif not recursive:
            self._delete()
            return
        elif self.is_dir():
            for child in self.iterdir():
                child.delete()
        else:
            self._delete()

    def _delete(self):
        # low level delete
        ...


class VFileSystem:

    def __init__(self):
        self._root: VPath = None

    @property
    def connected(self) -> bool:
        return self._root is not None

    def connect(self) -> Self:
        ...

    @property
    def root(self) -> VPath:
        assert self._root is not None
        return self._root

    def disconnect(self):
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False