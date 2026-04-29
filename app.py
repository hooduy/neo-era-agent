import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="Universal AI Agent", layout="centered")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🔑 API Keys Missing! Please add GEMINI_API_KEY and GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. INITIAL SETUP POPUP ---
if not st.session_state.ready:
    st.title("🤖 Personal AI Assistant")
    with st.form("setup_form"):
        st.write("Paste your Google Drive Folder Link to begin.")
        folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
        if st.form_submit_button("Initialize Chat"):
            if folder_url:
                st.session_state.ready = True
                st.rerun()
            else:
                st.warning("Please provide a link.")

# --- 3. THE FAST CHAT INTERFACE ---
else:
    st.title("💬 Universal Intelligence")
    
    if st.sidebar.button("➕ New Chat"):
        st.session_state.ready = False
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # This is the 'Brain' setup
            llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
            
            # This line tells the AI to be human and handle Hindi
            full_prompt = f"You are a helpful human-like AI. Answer this query naturally: {user_input}"
            
            # --- THE FIX IS HERE ---
            ai_reply = llm.invoke(full_prompt)
            answer = ai_reply.content
            
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
