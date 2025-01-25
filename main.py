
from typing import Set
from backend.core import run_llm

import streamlit as st

from streamlit_chat import message

st.header("Langchain Documentation helper bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "bot_answer_history" not in st.session_state:
    st.session_state["bot_answer_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response ..."):
        generated_response = run_llm(query=prompt)
        sources =  set ([doc.metadata["source"] for doc in generated_response["source_documents"]])
        gen_result = generated_response["result"]
        formatted_response = (
            f"{gen_result} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["bot_answer_history"].append(formatted_response)

if st.session_state["bot_answer_history"]:
    import pandas as pd

    df = pd.DataFrame({"col1" : ["row1", "row2", "row3"],
                       "col2" : ["row1", "row2", "row3"]})
    for user_query, generate_response in zip(st.session_state["user_prompt_history"], st.session_state["bot_answer_history"]):
        st.write(user_query)
        with st.chat_message("ai"):
            st.write(generate_response)
            st.dataframe(df)


