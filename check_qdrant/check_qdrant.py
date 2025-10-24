import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = qdrant_client.QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    collection_name="example_collection", client=client
)

print(vector_store.client.get_collections())
