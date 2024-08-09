from typing import Optional, cast


class TNode:
    def __init__(self, name="start", parent: Optional["TNode"] = None):
        self.name = name
        self.data: dict = {}
        self.parent = parent
        self.children: dict[str, "TNode"] = {}
        if parent is not None:
            parent.children[self.name] = self


class Tree:

    def __init__(self):
        self.root: TNode = TNode()

    def add_path(self, path: list[str]):
        n: TNode = cast(TNode, self.root)
        for p in path + ["end"]:
            if p not in n.children:
                c = TNode(p, n)
            else:
                c = n.children[p]
            n = c
        pass
    # end
# end

