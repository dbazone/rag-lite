import os
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

chroma_path = "data/chroma_db"
openai_key = os.getenv("OPENAI_API_KEY")


def cleanup_folder(directory):
    """Removes empty folders and .tmp files from the given directory."""
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

cleanup_folder(chroma_path)


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"   #https://platform.openai.com/docs/models#embeddings
)
chroma_client = chromadb.PersistentClient(path=chroma_path)
openai_client = OpenAI(api_key=openai_key)


collection_name = "docs"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)




def read_documents(directory_path):
    print("Reading Documents")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the directory
directory_path = "./data/documents"
documents = read_documents(directory_path)
print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# print(doc["embedding"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )
