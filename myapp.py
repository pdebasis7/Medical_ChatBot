from langchain.chat_models import AzureChatOpenAI
import openai
import streamlit as st
from src.helper import load_pdf, text_splitter, txt_embeddings
from src.prompt import *
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from store_index import vectordb, extracted_data, text_chunks, embeddings
import time

vectordb.persist()
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}



llm=AzureChatOpenAI( 
    #deployment_name=DEPLOYMENT_NAME,
    #model_kwargs={'deployment_name':'omsgenai1'},
    deployment_name='omsgenai1',
    openai_api_key='1ea71466d8ac49f4999b1984ffbab0a1',
    openai_api_version="2023-07-01-preview",
    openai_api_base='https://omsgenai.openai.azure.com/',
    temperature=0.5

)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

st.title("Medical Chat Bot")

if 'messages' not in st.session_state:
                # Start with first message from assistant
                st.session_state['messages'] = [{"role": "assistant", 
                                            "content": "Hi, How can I help you today?"}]
 
            # Display chat messages from history on app rerun
            # Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=None):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=None):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = qa({"query": query})
        response = result['result']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})