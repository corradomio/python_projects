from pathlib import Path
from typing import Generator, cast

from ftputil import FTPHost
from ftputil.file import FTPFile

from ..local import LocalPath
from ..utils import _split_url, parent_of, join_with
from ..vfs import VFileSystem, VPath, OPEN_MODES


# ---------------------------------------------------------------------------
# FTPPath
# ---------------------------------------------------------------------------

class FTPPath(VPath):
    def __init__(self, fs: FTPFileSystem, path: str):
        super().__init__(path)
        self.fs: FTPFileSystem = fs
        self.client: FTPHost = self.fs.client

    def exists(self) -> bool:
        return self.client.path.exists(self.path)

    def is_file(self) -> bool:
        return self.client.path.isfile(self.path)

    def is_dir(self) -> bool:
        return self.client.path.isdir(self.path)

    def iterdir(self) -> Generator['FTPPath', None, None]:
        if not self.exists(): return
        if not self.is_dir(): raise NotADirectoryError(self.path)

        content = self.client.listdir(self.path)
        for file in content:
            file_path = self.client.path.join(self.path, file)
            yield FTPPath(self.fs, file_path)
        pass

    @property
    def parent(self) -> FTPPath:
        if self.path == self.fs.root.path:
            return self
        parent_path = parent_of(self.path)
        return FTPPath(self.fs, parent_path)

    def __truediv__(self, child: str) -> 'FTPPath':
        assert isinstance(child, str)
        assert child != ".." and child != "."
        child_path = join_with(self.path, child)
        return FTPPath(self.fs, child_path)

    def _mkdir(self):
        self.client.mkdir(self.path)

    def _mkfile(self):
        if self.exists(): return
        file: FTPFile = self.client.open(self.path, mode='w')
        file.close()

    def _delete(self):
        if not self.exists():
            return
        elif self.is_dir():
            self.client.rmdir(self.path)
        else:
            self.client.remove(self.path)

    def open(self, mode: OPEN_MODES, **kwargs):
        return self.client.open(self.path, mode, **kwargs)


    def _copy(self, dest: VPath):
        # file (remote) -> file (remote)
        assert isinstance(dest, FTPPath)

        fs = None
        fd = None
        try:
            fs: FTPFile = cast(FTPFile, self.open('rb'))
            fd: FTPFile = cast(FTPFile, dest.open('wb'))
            self.client.copyfileobj(fs, fd)
        finally:
            if fs is not None: fs.close()
            if fd is not None: fd.close()
    # end

    def _copyfrom(self, src: Path):
        # file (remote) <- file (local)
        self.client.upload(str(src), self.path)

    def _copyto(self, dst: Path):
        # file (remote) -> file (local)
        self.client.download(self.path, str(dst))

    def open(self, mode='r', **kwargs):
        self.client.open(self.path, mode, **kwargs)
# end



# ---------------------------------------------------------------------------
# FTPFileSystem
# ---------------------------------------------------------------------------

class FTPFileSystem(VFileSystem):
    def __init__(self, url: str, username: str, password: str, **kwargs):
        super().__init__()
        assert url.startswith("ftp://")
        self._url = url
        self._username = username
        self._password = password
        self._kwargs = kwargs
        self.client: FTPHost = None

    def connect(self):
        assert self._root is None

        host, path = _split_url(self._url)

        ftp = FTPHost(host, self._username, self._password, **self._kwargs)
        # ftp.login(user=self._username, passwd=self._password, **self._kwargs)
        ftp.chdir(path)
        root_path = ftp.getcwd()
        self.client = ftp
        self._root = FTPPath(self, root_path)
        return self

    def disconnect(self):
        if self.client is not None:
            self.client.close()
        self.client = None
        self._root = None
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
