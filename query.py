import os
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

chroma_path = "data/chroma_db"
openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"   #https://platform.openai.com/docs/models#embeddings
)
chroma_client = chromadb.PersistentClient(path=chroma_path)
openai_client = OpenAI(api_key=openai_key)

collection_name = "docs"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer



question = input("Ask a question: ")
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer.content)