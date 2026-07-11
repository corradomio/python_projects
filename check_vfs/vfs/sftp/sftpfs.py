import stat
from pathlib import Path
from typing import Generator

from paramiko import SFTPClient, AutoAddPolicy, SSHClient, SFTPFile
from paramiko.sftp_attr import SFTPAttributes

from ..utils import _split_url, parent_of, join_with
from ..vfs import VFileSystem, VPath, VStream


# ---------------------------------------------------------------------------
# SFTPPath
# ---------------------------------------------------------------------------

class SFTPPath(VPath):
    def __init__(self, fs: SFTPFileSystem, path: str):
        super().__init__(path)
        self.fs: SFTPFileSystem = fs
        self.client = self.fs.client
        self._attrs: SFTPAttributes = None

    def exists(self) -> bool:
        try:
            return stat.S_ISDIR(self._stat().st_mode) or stat.S_ISREG(self._stat().st_mode)
        except FileNotFoundError:
            return False

    def is_dir(self) -> bool:
        return stat.S_ISDIR(self._stat().st_mode)

    def is_file(self) -> bool:
        return stat.S_ISREG(self._stat().st_mode)

    def iterdir(self) -> Generator['SFTPPath', None, None]:
        if not self.exists(): return
        if not self.is_dir(): raise NotADirectoryError(self.path)

        content = self.fs.client.listdir(self.path)
        for file in content:
            file_path = join_with(self.path, file)
            yield SFTPPath(self.fs, file_path)
        pass

    @property
    def parent(self) -> SFTPPath:
        if self.path == self.fs.root.path:
            return self
        parent_path = parent_of(self.path)
        return SFTPPath(self.fs, parent_path)

    def __truediv__(self, child: str) -> 'SFTPPath':
        assert isinstance(child, str)
        assert child != ".." and child != "."
        child_path = join_with(self.path, child)
        return SFTPPath(self.fs, child_path)

    def _mkdir(self):
        self.client.mkdir(self.path)

    def _mkfile(self):
        if self.exists(): return
        file: SFTPFile = self.client.file(self.path, mode='w')
        file.close()

    def _delete(self):
        if not self.exists():
            return
        elif self.is_dir():
            self.client.rmdir(self.path)
        else:
            self.client.remove(self.path)

    def _stat(self) -> SFTPAttributes:
        if self._attrs is None:
            self._attrs: SFTPAttributes = self.client.stat(self.path)
        return self._attrs

    def _copy(self, dest: VPath):
        # file (remote) -> file (remote)
        assert isinstance(dest, SFTPPath)
        pass

    def _copyfrom(self, src: Path):
        self.client.put(str(src), self.path)

    def _copyto(self, dst: Path):
        self.client.get(self.path, str(dst))

    def open(self, mode='r', buffer_size=0, **kwargs):
        buffer_size = buffer_size or SFTPFile.MAX_REQUEST_SIZE
        self.client.open(self.path, mode, bufsize=buffer_size, **kwargs)
# end


# ---------------------------------------------------------------------------
# SFTPFileSystem
# ---------------------------------------------------------------------------

class SFTPFileSystem(VFileSystem):
    def __init__(self, url: str, username: str, password: str, **kwargs):
        super().__init__()
        assert url.startswith("sftp://")
        self._url = url
        self._username = username
        self._password = password
        self._kwargs = kwargs
        self.client: SFTPClient = None

    def connect(self):
        assert self._root is None

        host, path = _split_url(self._url)
        port = 22

        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            hostname=host, port=port, username=self._username, password=self._password,
            look_for_keys=False,  # Stops Paramiko from scanning ~/.ssh/
            allow_agent=False  # Stops Paramiko from checking ssh-agent
        )
        sftp: SFTPClient = ssh.open_sftp()
        sftp.chdir(path)
        path = sftp.getcwd()
        self.client = sftp
        self._root = SFTPPath(self, path)
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
