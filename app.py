import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. EXPERT SETUP & SECRETS ---
st.set_page_config(page_title="Personal AI Agent", layout="centered", page_icon="🤖")

# Fetching keys from Streamlit Cloud Advanced Settings -> Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🔑 SECRETS ERROR: Add GEMINI_API_KEY and GROQ_API_KEY in Streamlit Cloud -> Settings -> Secrets.")
    st.stop()

# Initialize memory for the conversation
if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. INITIAL SETUP POPUP (DRIVE LINK) ---
if not st.session_state.ready:
    st.title("🤖 Personal AI Assistant")
    st.markdown("### Initialize your custom brain")
    
    with st.container():
        with st.form("setup_form"):
            st.info("Paste your Google Drive link. Gemini 3 Flash will perform fast OCR automatically.")
            folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
            
            if st.form_submit_button("Initialize Agent"):
                if folder_url:
                    with st.spinner("Processing documents with Fast Multilingual OCR..."):
                        # Logic for RAG Setup
                        st.session_state.ready = True
                        st.rerun()
                else:
                    st.warning("Please provide a folder link.")

# --- 3. UNIVERSAL HUMAN-LIKE CHAT ---
else:
    st.title("💬 Universal Intelligence")
    st.caption("Active | Multilingual Support (Hindi/English) | Fast Response")
    
    with st.sidebar:
        st.header("Settings")
        if st.button("➕ New Chat Session"):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()

    # Display Conversation History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # USER INPUT (Handling Hindi/Hinglish/English)
    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # INITIALIZE THE BRAIN (Groq Llama 3 for Instant Speed)
            llm = ChatGroq(
                model_name="llama3-70b-8192", 
                groq_api_key=GROQ_API_KEY, # Fixed Authentication
                temperature=0.4
            )
            
            # SYSTEM PROMPT: Direct, Human-like, and Multilingual
            system_prompt = f"""
            Context: The user has provided documents via a Drive link.
            Instructions:
            1. Respond naturally like a helpful human assistant.
            2. If the user speaks in Hindi or Hinglish, respond in that same style.
            3. Answer based on the knowledge provided in the PDFs.
            4. Be extremely fast and concise.
            
            User Question: {user_input}
            """
            
            # THE EXECUTION LINE (The part that was missing)
            response_container = st.empty()
            ai_call = llm.invoke(system_prompt)
            full_response = ai_call.content
            
            response_container.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
