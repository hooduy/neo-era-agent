import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# --- UI SETTINGS ---
st.set_page_config(page_title="NEO-ERA AI", layout="centered")

# --- SESSION STATE (Memory) ---
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_id(url):
    match = re.search(r'folders/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else url

# --- POPUP SCREEN ---
if not st.session_state.setup_complete:
    st.title("📂 Initialize Startup Agent")
    with st.form("startup_form"):
        st.write("Paste your Google Drive Folder Link to begin.")
        gdrive_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/drive/folders/...")
        
        # Keys provided via UI for first-time setup
        gemini_api = st.text_input("Gemini API Key (OCR):", type="password")
        groq_api = st.text_input("Groq API Key (Chat):", type="password")
        
        if st.form_submit_button("Start OCR & Chat"):
            if gdrive_url and gemini_api and groq_api:
                with st.spinner("Agent is reading PDFs using Gemini 3 Flash..."):
                    try:
                        # In a professional app, you'd use Google Drive API here. 
                        # For this prototype, we'll guide the user to provide 
                        # local access or use a simplified loader.
                        
                        st.session_state.gemini_key = gemini_api
                        st.session_state.groq_key = groq_api
                        st.session_state.setup_complete = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Setup Error: {e}")
            else:
                st.warning("All fields are required!")

# --- CHAT INTERFACE ---
else:
    st.title("💬 Chat with Mining Docs")
    
    if st.sidebar.button("New Chat"):
        st.session_state.setup_complete = False
        st.session_state.messages = []
        st.rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the extracted text..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # LangChain Agent Logic
            llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=st.session_state.groq_key)
            
            # This is where your OCR-extracted text would be queried
            response = "Agent Analysis: Based on the OCR scan of your folder, I found that..."
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
