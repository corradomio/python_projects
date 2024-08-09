from typing import Optional

import matplotlib.pyplot as plt
from collections import deque
import netx
import stdlib.csvx as csvx


# def add_iid(G, iid: str):
#     n = "start"
#
#     for c in iid:
#         G.add_edge(n, c)
#         n = c
#
#     G.add_edge(n, "end")
# # end

# isalnum
# isalpha
# isascii
# isdeciman
# isdigit
# islower
# isupper
# isidentifier
# isnumeric
# isprintable
# isspace
# istitle
# .


class RegexpDiscover:

    def __init__(self, name, ratio=0.5):
        self.name = name
        self.ratio = ratio
        T = netx.Tree()
        self.T = T

    def add(self, s: str):
        T = self.T

        T.add_path([c for c in s])

    # end

    def process_tree(self):
        N = self.T.root

        toprocess = deque([N])

        while len(toprocess) > 0:
            N = toprocess.pop()

            slist = list(N.children.keys())
            if 'end' in slist:
                slist.remove('end')

            replacement = self.analyze_successors(slist)
            if replacement is not None:
                children = {}
                for C in N.children:
                    children |= N.children[C].children

                N.name = replacement
                N.children = children
                toprocess.extend(children.values())
            else:
                toprocess.extend(N.children.values())
        pass
    # end

    def analyze_successors(self, slist):

        if len(slist) <= 1:
            return None

        ratio = self.ratio
        unique = set(slist)

        def is_digits(s):
            return all(map(lambda c: c.isdigit(), unique))

        def is_lower(s):
            return all(map(lambda c: c.islower(), unique))

        def is_upper(s):
            return all(map(lambda c: c.isupper(), unique))

        def is_alpha(s):
            return all(map(lambda c: c.isupper(), unique))

        if is_digits(unique):
            if len(unique) / 10 < ratio:
                return "[" + "".join(unique) + "]"
            else:
                return "\\d"

        if is_lower(unique):
            if len(unique) / 26 < ratio:
                return "[" + "".join(unique) + "]"
            else:
                "\\l"

        if is_upper(unique):
            if len(unique) / 26 < ratio:
                return "[" + "".join(unique) + "]"
            else:
                "\\u"

        if is_alpha(unique):
            if len(unique) / 26 < ratio:
                return "[" + "".join(unique) + "]"
            else:
                "\\a"

        return None
# end


def main():
    invoices = csvx.load_csv("generatedInvoices_v3.csv", skiprows=1, dtype=[None]*3 + [str, str])

    invoice_trees: dict[str, RegexpDiscover] = {}

    print("Generate trees ...")
    for invoice in invoices:
        i_id = invoice[0]
        name = invoice[1]

        if name not in invoice_trees:
            print(f"... {name}")
            rd = RegexpDiscover(name)
            invoice_trees[name] = rd
        else:
            rd = invoice_trees[name]

        rd.add(i_id)
        print(f"... ... {i_id}")
        # cnt += 1
        # if cnt == 10:
        #     break

    print("Process trees ...")
    for name in invoice_trees:
        print(f"... {name}")
        rd = invoice_trees[name]
        rd.process_tree()

        pass
    # print("Save trees ...")
    # for name in invoice_trees:
    #     print(f"... {name}")
    #     rd = invoice_trees[name]
    #     plt.title(name)
    #     netx.draw(rd.G)
    #     plt.savefig(f"plots/{name}.png", dpi=300)
    #     plt.close()

    print("Done")
    pass


if __name__ == "__main__":
    # logging.config.fileConfig('logging_config.ini')
    # log = logging.getLogger("root")
    # log.info("Logging system configured")
    main()

