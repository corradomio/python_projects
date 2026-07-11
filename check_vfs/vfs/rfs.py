from .utils import protocol_of
from .local import LocalFileSystem
from .ftp import FTPFileSystem
from .sftp import SFTPFileSystem

REGISTERED_FILE_SYSTEMS = {
    '':LocalFileSystem,
    'file://': LocalFileSystem,
    'ftp://':FTPFileSystem,
    'sftp://':SFTPFileSystem
}


def create(url: str, **kwargs):
    proto = protocol_of(url)
    if proto in REGISTERED_FILE_SYSTEMS:
        return REGISTERED_FILE_SYSTEMS[proto](url, **kwargs)
    raise NotImplementedError(url)
# end
