import os
# from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import uuid

# load_dotenv()

def pdf_path_to_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    langchain_documents = []
    for doc in documents:
        langchain_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "project": "project_1",
                },
            )
        )
    return langchain_documents

def split_pdf(pdf_path, chunk_size=800, chunk_overlap=400):
    langchain_documents = pdf_path_to_document(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(langchain_documents)
    return texts

pdf_path = "data/Raju Codes_ A Journey from Village Dreamer to Tech Hero.pdf"
chunks = split_pdf(pdf_path)
for chunk in chunks:
    print(chunk.metadata)

# ---------------------------------------------------------------------------

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION")
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", organization=OPENAI_ORGANIZATION
)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "stories"

# Defining the model
model = ChatOpenAI(model="gpt-3.5-turbo-0125", organization=OPENAI_ORGANIZATION)

# Defining the qdrant client
qdrant_client = QdrantClient(url=URL, prefer_grpc=True)

def add_unique_id(chunks):
    for chunk in chunks:
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['source']}_{chunk.metadata['page']}_{chunk.metadata['start_index']}"
        )
    return chunks

chunks = add_unique_id(chunks)

def get_or_create_collection(collection_name):
    if qdrant_client.collection_exists(collection_name=f"{COLLECTION_NAME}"):
        return collection_name
    return qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

def add_to_qdrant(chunks, embeddings):
    get_or_create_collection(COLLECTION_NAME)
    new_chunks = [
        chunk
        for chunk in chunks
        if not get_chunk_by_chunk_id(COLLECTION_NAME, chunk.metadata["chunk_id"])[0]
    ]
    if not new_chunks:
        print("No new chunks to add to Qdrant")
        return
    print(f"Adding {len(new_chunks)} new chunks to Qdrant")
    qdrant = Qdrant.from_documents(
        new_chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        url=URL,
        prefer_grpc=True,
    )
    return qdrant

vector_db = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=openai_embeddings,
)
add_to_qdrant(chunks, openai_embeddings)
print("Database: ", vector_db)
