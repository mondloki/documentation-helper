import os

from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_docs():

    loader = ReadTheDocsLoader("langchain-docs2/", encoding="ISO-8859-1")
    try:
        raw_documents = loader.load()
    except Exception as e:
        print(e)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(documents=raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs2", "https://api.python.langchain.com/en/latest/chains/")
        doc.metadata.update({"source": new_url})



    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # PineconeVectorStore.from_documents(documents, embedding_model, index_name=os.environ['INDEX_NAME'])

    vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embedding_model)
    vector_store.delete(delete_all=True)
    print("Data indexed...")

if __name__ == "__main__":
    ingest_docs()