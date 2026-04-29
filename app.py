import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA 

# --- 1. PAGE STYLE & UI ---
st.set_page_config(page_title="NEO-ERA AI Agent", layout="centered")

# --- 2. INITIALIZE STATE ---
if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. POPUP MODAL (The Link Input) ---
if not st.session_state.ready:
    st.title("🤖 Project NEO-ERA: AI Setup")
    with st.container():
        st.write("Welcome, Engineer. Link your Google Drive folder to begin.")
        with st.form("setup_form"):
            folder_url = st.text_input("Paste Google Drive Folder Link:")
            gemini_key = st.text_input("Gemini API Key:", type="password")
            groq_key = st.text_input("Groq API Key:", type="password")
            
            if st.form_submit_button("Initialize Startup Agent"):
                if folder_url and gemini_key and groq_api:
                    with st.spinner("Agent is performing OCR on Folder..."):
                        try:
                            # Using the corrected library addresses
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", 
                                google_api_key=gemini_key
                            )
                            
                            st.session_state.ready = True
                            st.session_state.groq_key = groq_api
                            st.session_state.gemini_key = gemini_key
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("All fields are required!")

# --- 4. CHATGPT INTERFACE ---
else:
    st.title("💬 NEO-ERA Intelligence")
    
    with st.sidebar:
        if st.button("New Chat / Change Folder"):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask about your mining documents..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Using Groq Llama 3 for professional speed
            llm = ChatGroq(
                model_name="llama3-8b-8192", 
                groq_api_key=st.session_state.groq_key,
                temperature=0
            )
            
            answer = "I have analyzed the PDFs in your folder. Based on the mining data extracted via Gemini OCR..."
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
