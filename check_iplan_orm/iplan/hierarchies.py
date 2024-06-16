from typing import Optional, Union, Literal
from stdlib import as_list
from sqlalchemy import delete, insert

from .common import *
from .ombase import *


# ---------------------------------------------------------------------------
# Attribute Hierarchy
#       Area  Hierarchy
#       Skill Hierarchy
# ---------------------------------------------------------------------------

class AttributeDetail(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.AttributeDetail)
        self.check_data()
        self.parent = None
        self.children = []

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def name(self) -> str:
        return self.data['attribute']

    @property
    def description(self) -> str:
        return self.data['description']

    @property
    def parent_id(self) -> Optional[int]:
        """For the root, 'parent_id' is None"""
        return self.data['parent_id']

    @property
    def hierarchy_id(self):
        return self.data['attribute_master_id']
# end


class AttributeHierarchy(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.AttributeMaster)
        self._name: Optional[str] = None
        self._type: Literal['area', 'skill'] = None

    def set_name_type(self, name: str, type: str):
        assert is_instance(name, str)
        assert is_instance(type, str)
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name

        self.check_data()
        return self.data['attribute_master_name']

    @property
    def type(self) -> Literal["area", "skill"]:
        if self._type is not None:
            return self._type

        self.check_data()
        type = self.data['hierarchy_type']
        if type == 1:
            return "area"
        elif type == 2:
            return "skill"
        else:
            raise ValueError(f"Unsupported hierarchy type {type}")

    @property
    def description(self) -> str:
        self.check_data()
        return self.data['attribute_desc']

    def details(self) -> list[AttributeDetail]:
        with self.engine.connect() as conn:
            table = self.ipom.AttributeDetail
            query = select(table).where(table.c['attribute_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query)#.fetchall()
            # idlist: [(id,), ...]
            return [AttributeDetail(self.ipom, to_data(res)) for res in rlist]

    # alias
    def features(self) -> list[AttributeDetail]:
        return self.details()

    def feature_ids(self, leaf_only=False, with_name=False) -> Union[list[int], dict[int, str]]:
        """
        Return the id's list of all hierarchy's nodes
        If it is specified 'with_name=True', it is returned the dictionary {id: name}

        :param leaf_only: if True, only the leaf nodes are returned
        :param with_name: if to return the dictionary {id: name}
        :return:
        """
        with self.engine.connect() as conn:
            table = self.ipom.AttributeDetail
            if leaf_only:
                query = select(table.c['id', 'attribute']).where(
                    (table.c['attribute_master_id'] == self.id) &
                    (table.c['is_leafattribute'] == True)       # WARN: DOESN'T change '== True' into 'is True'
                )
            else:
                query = select(table.c['id', 'attribute']).where(
                    (table.c['attribute_master_id'] == self.id)
                )

            self.log.debug(f"{query}")
            rlist = conn.execute(query)#.fetchall()
            # rlist: [(id, name), ...]
            if with_name:
                return {res[0]: res[1] for res in rlist}
            else:
                return [rec[0] for rec in rlist]

    def tree(self) -> AttributeDetail:
        root = None
        nodes = self.details()
        node_dict = {}
        for node in nodes:
            node_dict[node.id] = node

        for node in nodes:
            parent_id = node.parent_id
            if parent_id is None:
                root = node
                continue
            parent = node_dict[parent_id]
            node.parent = parent
            parent.children.append(node)
        return root

    def to_ids(self, attr: Union[None, int, list[int], str, list[str]],
               leaf_only=False,
               with_name=False) -> Union[list[int], dict[int, str]]:
        """
        Convert the attribute(s) in a list of attribute ids.
        The attribute can be specified as id (an integer) or name (a string)
        If attr is None, all leaf attributes are returned

        :param attr: attribute(s) to convert
        :param with_name: if to return
        :param leaf_only: if to select the leaf attributes
        :return: list of attribute ids
        """
        feature_dict = self.feature_ids(leaf_only=leaf_only, with_name=True)
        feature_drev = reverse_dict(feature_dict)
        aids = []

        # if attr is None, return all feature ids
        if attr is None:
            if with_name:
                return feature_dict
            else:
                return list(feature_dict.keys())
        # end

        attrs = as_list(attr)
        attr_ids = []
        for aid in attrs:
            if isinstance(aid, int):
                assert aid in feature_dict, f"Invalid attribute {aid}: not available in hierarchy {self.name}"
                attr_ids.append(aid)
            elif is_instance(aid, str):
                assert aid in feature_drev, f"Invalid attribute {aid}: not available in hierarchy {self.name}"
                attr_ids.append(feature_drev[aid])
            else:
                raise ValueError(f"Invalid attribute type {type(attr)}: unsupported in in hierarchy {self.name}")

        if with_name:
            return {
                aid: feature_dict[aid]
                for aid in attr_ids
            }
        else:
            return aids
    # end

    # -----------------------------------------------------------------------

    def delete(self):
        if self.id == NO_ID:
            return self

        self._name = self.name
        self._type = self.type

        self._delete_attribute_hierarchy(self.id, self.type)

        super().delete()
        return self
    # end

    def _delete_attribute_hierarchy(self, area_hierarchy_id: int, hierarchy_type: Literal['area', 'skill']):
        with self.engine.connect() as conn:
            # 1) delete dependencies
            # ...

            # 2) delete tb_attribute_details
            table = self.ipom.AttributeDetail
            query = delete(table).where(table.c['attribute_master_id'] == area_hierarchy_id)
            self.logsql.debug(query)
            conn.execute(query)

            # 3) delete tb_attribute_master
            table = self.ipom.AttributeMaster
            query = delete(table).where(table.c.id == area_hierarchy_id)
            self.logsql.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def create(self, hierarchy_tree):
        if self._id != NO_ID:
            self.log.warning(f"{self._type} Hierarchy '{self._name}' already existent")
            return self

        assert is_instance(self._name, str), "Missing hierarchy name"
        assert is_instance(self._type, str), "Missing hierarchy type"

        if len(hierarchy_tree) == 1 and is_instance(hierarchy_tree, dict[str, list[str]]):
            self._id = self._create_simple_hierarchy(self._name, self._type, hierarchy_tree)
        else:
            raise ValueError(f"Unsupported hierarchy tree format: {hierarchy_tree}")

        self._name = None
        self._type = None
        return self
    # end

    def _create_simple_hierarchy(self, name: str, hierarchy_type: Literal['area', 'skill'], hierarchy_tree: dict[str, list[str]]) -> int:
        now = datetime.now()
        root_name = list(hierarchy_tree.keys())[0]
        leaf_names = hierarchy_tree[root_name]
        description = name

        # hierarchy_tree:
        #   {parent: list[Union[str, dict[str, list]]}
        #   {child: parent}

        with self.engine.connect() as conn:
            # 1) create tb_attribute_master
            table = self.ipom.AttributeMaster
            query = insert(table).values(
                attribute_master_name=name,
                attribute_desc=description,
                createdby=CREATED_BY,
                createddate=now,
                hierarchy_type=1 if hierarchy_type == 'area' else 2
            ).returning(table.c.id)
            self.logsql.debug(query)
            hierarchy_id = conn.execute(query).scalar()
            # 2) create tb_attribute_detail
            #    simple format: {root: list[leaf]}
            table = self.ipom.AttributeDetail
            # 2.1) create the root
            query = insert(table).values(
                attribute_master_id=hierarchy_id,
                attribute=root_name,
                description=root_name,
                attribute_level=1,
                parent_id=None,
                createdby=CREATED_BY,
                createddate=now,
                is_leafattribute=False
            ).returning(table.c.id)
            self.logsql.debug(query)
            parent_id = conn.execute(query).scalar()
            # 2.2) create the leaves
            for leaf in leaf_names:
                query = insert(table).values(
                    attribute_master_id=hierarchy_id,
                    attribute=leaf,
                    description=leaf,
                    attribute_level=2,
                    parent_id=parent_id,
                    createdby=CREATED_BY,
                    createddate=now,
                    is_leafattribute=True
                ).returning(table.c.id)
                leaf_id = conn.execute(query).scalar()
            # end
            conn.commit()
        # return AttributeHierarchy(self, hierarchy_id, self.AttributeMaster)
        return hierarchy_id
    # end
# end


class PeriodHierarchy(IPlanObject):

    SUPPORTED_FREQS = {
        'day': 'D',     # day start
        'week': 'W',    # week start
        'month': 'M'    # month start
    }

    def __init__(self, ipom, period_hierarchy, period_length):
        super().__init__(ipom)
        assert period_hierarchy in self.SUPPORTED_FREQS

        # period_hierarchy
        self._period_hierarchy = period_hierarchy
        self._period_length = period_length

    @property
    def freq(self) -> Literal['D', 'W', 'M']:
        """Frequency Pandas's compatible"""
        return self.SUPPORTED_FREQS[self._period_hierarchy]

    @property
    def periods(self) -> int:
        return self._period_length

    # def date_range(self, start=None, end=None, periods=None) -> pd.DatetimeIndex:
    #     assert is_instance(start, Union[None, datetime])
    #     assert is_instance(end, Union[None, datetime])
    #     assert is_instance(periods, Union[None, int])
    #     return pd.date_range(start=start, end=end, periods=periods, freq=self.freq)

    # def period_range(self, start=None, end=None, periods=None) -> pd.PeriodIndex:
    #     assert is_instance(start, Union[None, datetime])
    #     assert is_instance(end, Union[None, datetime])
    #     assert is_instance(periods, Union[None, int])
    #     return pd.period_range(start=start, end=end, periods=periods, freq=self.freq)

    def __repr__(self):
        return f"{self._period_hierarchy}:{self._period_length}"
# end


class AttributeHierarchies(IPlanObject):

    def __init__(self, ipom):
        super().__init__(ipom)
        self.AttributeMaster = self.ipom.AttributeMaster
        self.AttributeDetail = self.ipom.AttributeDetail

    def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self._attribute_hierarchy(id, "area")

    def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self._attribute_hierarchy(id, "skill")

    def _attribute_hierarchy(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
            -> AttributeHierarchy:
        hierarchy_id = self._convert_id(
            id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'], nullable=True
        )

        if hierarchy_id is None:
            hierarchy = AttributeHierarchy(self.ipom, NO_ID)
            hierarchy.set_name_type(str(id), hierarchy_type)
        else:
            hierarchy = AttributeHierarchy(self.ipom, hierarchy_id)
        assert hierarchy.type == hierarchy_type, f"Invalid hierarchy {id}: required '{hierarchy_type}', found '{hierarchy.type}'"
        return hierarchy

    def attribute_detail(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
            -> AttributeDetail:
        feature_id = self.ipom._convert_id(id, self.ipom.AttributeDetail, ['attribute', 'description'])
        detail = AttributeDetail(self.ipom, feature_id)
        assert self.hierarchy_type(detail.hierarchy_id) == hierarchy_type
        return detail

    def hierarchy_type(self, id: Union[int, str]) -> Literal["area", "skill"]:
        hierarchy_id = self._convert_id(id, self.ipom.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

        with self.engine.connect() as conn:
            table = self.AttributeMaster
            query = select(table.c['hierarchy_type']).where(table.c['id'] == hierarchy_id)
            self.log.debug(f"{query}")
            hierarchy_type = conn.execute(query).fetchone()[0]
        return "area" if hierarchy_type == 1 else "skill"
# end
