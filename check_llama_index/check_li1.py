from llama_index.core import Document

text = "The quick brown fox jumps over the lazy dog."
doc = Document(
    text=text,
    metadata={'author': 'John Doe', 'category': 'others'},
    id_='1'
)
print(doc)

# % --

from llama_index.readers.wikipedia import WikipediaReader

loader = WikipediaReader()
documents = loader.load_data(
    pages=['Pythagorean theorem', 'Physicist', 'London']
    # pages=['Physicist']
)
print(f"loaded {len(documents)} documents")
