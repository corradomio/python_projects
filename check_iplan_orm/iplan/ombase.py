import stdlib.loggingx as logging
from typing import Union, Any

from sqlalchemy import Row
from sqlalchemy import Table
from sqlalchemy import select

from stdlib import is_instance


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

CREATED_BY = "Python IPlanObjectModel"
NO_ID = -1
NO_NAME = "<noname>"


def to_data(result: Row) -> dict[str, Any]:
    values = result._data
    fields = result._fields
    assert len(values) == len(fields)
    return {
        fields[i]: values[i] for i in range(len(result))
    }


# ---------------------------------------------------------------------------
# IPlanObject
#     IPlanData
# ---------------------------------------------------------------------------

class IPlanObject:

    def __init__(self, ipom):
        self.ipom: "IPlanObjectModel" = ipom
        self.engine = ipom.engine
        self._convert_id = ipom._convert_id

        self.log = logging.getLogger(f"iplan.om.{self.__class__.__name__}")
        self.logsql = logging.getLogger(f"iplan.om.sql.{self.__class__.__name__}")
    # end
# end


class IPlanData(IPlanObject):
    def __init__(self, ipom, id: Union[int, dict], table: Table, idcol="id"):
        super().__init__(ipom)
        self._table = table
        self._idcol = idcol

        self._id: int = NO_ID
        self._name: str = None
        self._data: dict[str, Any] = {}

        if isinstance(id, int):
            self._id = id
            self._data: dict[str, Any] = {}
        elif isinstance(id, dict):
            self._data = id
            self._id = self.data[idcol]
        elif is_instance(id, str):
            self._name = id
        else:
            raise ValueError(f"Unsupported {id}")

        self.log = logging.getLogger(f"iplan.om.{self.__class__.__name__}")

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def idcol(self) -> str:
        return self._idcol

    @property
    def table(self) -> Table:
        return self._table

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def check_data(self):
        if len(self._data) > 0:
            return

        with self.engine.connect() as conn:
            table = self._table
            query = select(table).where(table.c[self._idcol] == self._id)
            self.log.debug(f"{query}")
            result = conn.execute(query).fetchone()
            self._data = to_data(result)

    # -----------------------------------------------------------------------

    def exists(self) -> bool:
        return self._id != NO_ID

    def delete(self):
        self._id = NO_ID
        self._data = {}
        return self

    def create(self, **kwargs):
        return self

    def __repr__(self):
        return f"{self.name}:{self.id}"
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
