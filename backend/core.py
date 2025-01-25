import os

from dotenv import load_dotenv

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain import hub


def run_llm(query:str):

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    # other params...
)


    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embedding_model)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # retrieval_qa_chat_prompt = hub.pull("rlm/rag-prompt")

    template = """
human

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {input} 

Context: {context} 

Answer:
"""

    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template)

    comb_doc_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(),
                                             combine_docs_chain=comb_doc_chain)
    
    result = retrieval_chain.invoke(input={"input" : query})


    new_result = {
        "query" : result["input"],
        "result" : result["answer"],
        "source_documents" : result["context"]
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is langchain chain")
    print(res)