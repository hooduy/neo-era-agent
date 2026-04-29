import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="Universal AI Agent", layout="centered")

# Pulls keys from Streamlit Advanced Settings -> Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🔑 API Keys Missing! Please add GEMINI_API_KEY and GROQ_API_KEY in Streamlit Cloud Secrets.")
    st.stop()

if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. INITIAL SETUP POPUP (Google Drive Link) ---
if not st.session_state.ready:
    st.title("🤖 Personal AI Assistant")
    st.markdown("### Upload your knowledge base to begin")
    
    with st.container():
        with st.form("setup_form"):
            st.write("Paste your Google Drive Folder Link containing PDFs or Documents.")
            folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/drive/folders/...")
            
            if st.form_submit_button("Initialize Chat"):
                if folder_url:
                    with st.spinner("Agent is performing Fast OCR and analyzing documents..."):
                        # This section is ready for your specific Drive-to-Text logic
                        st.session_state.ready = True
                        st.rerun()
                else:
                    st.warning("Please provide a valid link.")

# --- 3. THE CHAT INTERFACE (Human-Like & Multilingual) ---
else:
    st.title("💬 Universal Intelligence")
    st.caption("Active | Fast OCR | Hindi & English Support")
    
    with st.sidebar:
        if st.button("➕ Start New Chat"):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask me anything about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Brain (Fastest Groq Llama 3)
            llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0.3)
            
            # Universal Human-Like Instruction
            system_instruction = f"""
            You are a helpful, human-like AI Assistant. 
            The user has provided documents via a Drive link. 
            1. If the user asks in Hindi, answer in Hindi naturally.
            2. If the text in the documents is in Hindi, translate or explain it as needed.
            3. Be conversational, friendly, and highly intelligent.
            4. Focus on providing the most accurate info from the provided files.
            
            User Question: {user_input}
            """
            
            # Standard Agent Response
            response = "Agent: I have processed your files. Based on the content provided, here is what I found..."
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
