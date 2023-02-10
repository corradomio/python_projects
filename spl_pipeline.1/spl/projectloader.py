import logging
import json
from logging import Logger
from typing import Optional, Any, Iterator
from path import Path
from commons import as_path


def source_file_extension(programming_language: str):
    if programming_language == 'java':
        return ".java"
    if programming_language in ["c#", "csharp"]:
        return ".cs"
    if programming_language == "python":
        return ".py"
    else:
        raise ValueError(f"Programming language '{programming_language}' not supported")


class ProjectLoader:
    def __init__(self, programming_language: str = 'java'):
        """
        Load a project defined in the json file '.spl/
        :param programming_language: programming language
        """
        self._home: Path = as_path(".")
        self._revision: int = 0
        self._refid: Optional[str] = None
        self._file_ext: str = source_file_extension(programming_language)
        self._files: list[dict] = []
        self._log = logging.getLogger("ProjectLoader")
    # end

    @property
    def home(self):
        return self._home
    # end

    @property
    def files(self) -> Iterator[str]:
        return map(lambda fi: fi["file"], self._files)
    # end

    def load(self, home: str, rev: int = 0, refid: Optional[str] = None):
        """
        Load the files from the specified directory of 'source-project' configuration file
        :param home: project home
        :param rev: project revision (default 0)
        :param refid: project reference id (default None)
        :return:
        """
        self._log.info(f"Scan {home} ...")
        self._home = as_path(home)
        self._rev = rev
        self._refid = refid

        source_project_file = self._compose_source_project_file_path()
        self._load_source_files(source_project_file)

        self._log.info(f"End {len(self._files)}")
    # end

    def _compose_source_project_file_path(self) -> Optional[Path]:
        """
        Find the first '[refid]-source-project-r[rev].json' file.
        If 'refid' is not specified, it is used the first file found
        If no file is found, it returns None

        :return: the founded file or None
        """
        source_project_file: Optional[str] = None

        spl_dir: Path = Path(self._home).joinpath(".spl")
        if not spl_dir.exists():
            return source_project_file

        rev = self._rev
        refid = self._refid
        for f in spl_dir.files(f"*-source-project-r{rev:02}.json"):
            if refid is not None and f.startswith(refid):
                source_project_file = f
                break
            elif refid is not None:
                continue
            else:
                source_project_file = f
        # end
        return source_project_file
    # end

    def _load_source_files(self, source_project_file: Optional[Path]):
        """
        Load the source files specified in 'source_project_file' JSON file
        or all files with the extension
        :param source_project_file:
        """
        if source_project_file is not None and source_project_file.exists():
            self._load_from_source_project_file(source_project_file)
        else:
            self._load_all_files()
    # end

    def _load_from_source_project_file(self, source_project_file: Path):
        """
        Load the source files defined in the configuration file
        :param source_project_file:
        """
        phome = self._home
        with open(source_project_file) as jrdr:
            jproject = json.load(jrdr)
            modules: dict[str, Any] = jproject["modules"]
            for module in modules:
                sources = modules[module]["sources"]
                for source in sources:
                    rpath = sources[source]["path"]
                    file = phome.joinpath(rpath)
                    assert file.exists()
                    finfo = {"path": rpath, "file": file}
                    self._files.append(finfo)
            pass
    # end

    def _load_all_files(self):
        """
        Scan the directory and retrieve all files with the selected extension
        """
        self._files = []
        file_ext = self._file_ext
        sel_files = "*" + self._file_ext
        for dir in self._home.walkdirs():
            # skip directories starting with "."
            if "\\." in dir or "/." in dir:
                continue
            # skip directories 'test'
            if "\\test\\" in dir or "/test/" in dir:
                continue

            for file in dir.files(sel_files):
                assert file.name.endswith(file_ext)
                self._files.append(file)
            # end
        # end

        pass
    # end

# end
