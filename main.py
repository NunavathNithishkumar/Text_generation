import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import openai
import io
import pickle
import os
import numpy as np
def get_pdf_text(path):
    loader = PyPDFLoader(path)
    raw_documents = loader.load()
    return raw_documents

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
            )    
    
st.title("Text Generation Web App :fire:")
st.subheader("Welcome to the chat")


with st.sidebar:
    st.title("Type your API-Key")
    text1=st.text_input("API Key")
    button1=st.button("Use Key")
    if button1:
      openai.api_key = text1
      st.sidebar.success("Done")

    raw_documents=get_pdf_text("48lawsofpower.pdf")

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    chunks = text_splitter.split_documents(raw_documents)
    embedding_model_id = "BAAI/bge-small-en-v1.5"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
    )
    # Saving Embeddings in VectorStore
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
            
vector_store = FAISS.load_local("faiss_index", embeddings)  
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.write("""
#### Ask your Question.
 """)
q1=st.text_area("Write your Question here.")
but1=st.button('Submit')
question=""

#openai.api_key = apikey

res=""
if but1:
    question=q1
    #vector_store= vector_store.similarity_search(question)
    system_template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}

    Begin!
    ----------------
    Question: {question}
    Helpful Answer:"""

    
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)



    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key = openai.api_key)

    from langchain.chains import RetrievalQA
  
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs = {"prompt": prompt})
    result = qa_chain({"query": question })
   
    st.write(result)

    
    
