from pprint import pprint

import networkx as nx
from pypdf import PdfReader


def read_book():

    reader = PdfReader("murphy.pdf")
    n_pages = len(reader.pages)
    body = ""
    for p in range(n_pages):
        page = reader.pages[p]
        body += page.extract_text() + "\n"
    # end
    return body


def filter_book(book: str):
    lines = book.split("\n")
    sentences = [
        line
        for line in lines
        if line.startswith("    ") and not line.startswith("                              ")
    ]
    return sentences

UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"

def concat_lines(lines: list):
    clines = []
    for line in lines:
        line = line.strip()
        if line[0] in UPPERCASE:
            clines.append(line)
        elif line.endswith(":"):
            clines[-1] = clines[-1] + " " + line
        elif line[0] in LOWERCASE:
            clines[-1] = clines[-1] + " " + line
        else:
            clines.append(line)
    # end
    return clines


def lower_lines(lines: list):
    llines = []
    for line in lines:
        line = line.strip().lower()
        llines.append(line)
    return llines


class WordDict:
    def __init__(self):
        self._w2i: dict[str, int] = {}
        self._i2w: dict[int, str] = {}

    def __getitem__(self, item: str|int):
        if isinstance(item, str):
            return self._w2i[item]
        else:
            return self._i2w[item]

    def index(self, w: str) -> int:
        w = w.lower()
        if w not in self._w2i:
            i = len(self._w2i)
            self._w2i[w] = i
            self._i2w[i] = w
        return self._w2i[w]

    def word(self, i: int) -> str:
        return self._i2w[i]

    def list_(self, sentence: list[str]) -> list[int]:
        return [
            self.index(w)
            for w in sentence
        ]

    def sentence(self, il: list[int]) -> list[str]:
        return [
            self.word(i)
            for i in il
        ]
# end


def main():
    book = read_book()
    lines = filter_book(book)
    lines = concat_lines(lines)
    lines = lower_lines(lines)
    for line in lines:
        print(line)

    lines = lines[0:10]

    D = WordDict()
    G = nx.DiGraph()

    # for sentence in lines:
    #     words = sentence.split()
    #     il = D.list_(words)
    #     n1 = len(il)-1
    #
    #     for i in range(n1):
    #         j = i+1
    #         u = il[i]
    #         v = il[j]
    #         G.add_node(u, word=D[u])
    #         G.add_node(v, word=D[v])
    #         G.add_edge(u,v)
    #     pass
    # # end

    for sentence in lines:
        words = sentence.split()
        n1 = len(words)-1

        for i in range(n1):
            j = i+1
            u = words[i]
            v = words[j]
            G.add_node(u)
            G.add_node(v)
            G.add_edge(u,v)
        pass
    # end

    nx.write_gml(G, "murphy.gml")
    nx.write_graphml(G, "murphy.graphml")
    pass


if __name__ == "__main__":
    main()
