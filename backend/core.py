import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.globals import set_debug
set_debug(True)

from langchain.chains.history_aware_retriever import create_history_aware_retriever

def run_llm(query:str,
            chat_history: List[Dict[str, Any]]):

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

You are an assistant chatbot for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {input} 

chat history : {chat_history_str}

Context: {context} 

Answer:
"""
    ### Include chat history in prompt ###
    chat_history_str = ""
    for history in chat_history:
        chat_history_str += f"{history[0]} : {history[1]}"
        chat_history_str += "\n"
    ### Include chat history in prompt ###
    
    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template, partial_variables={"chat_history_str" : chat_history_str})

    stuff_documents_chain  = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)

    ### Include chat history in vector search ###
    history_rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                             retriever=vector_store.as_retriever(),
                                                             prompt=history_rephrase_prompt)
    ### Include chat history in vector search ###
    retrieval_chain = create_retrieval_chain(retriever=history_aware_retriever,
                                             combine_docs_chain=stuff_documents_chain)
    

    
    result = retrieval_chain.invoke(input={"input" : query,
                                           "chat_history" : chat_history})


    new_result = {
        "query" : result["input"],
        "result" : result["answer"],
        "source_documents" : result["context"]
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is langchain chain")
    print(res)