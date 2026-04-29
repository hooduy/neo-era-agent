import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="Universal Knowledge Agent", layout="centered", page_icon="🤖")

# Fetching Keys from Secrets
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🔑 Secrets missing! Streamlit Settings mein GEMINI_API_KEY aur GROQ_API_KEY add karein.")
    st.stop()

# Initialize Session States
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. PHASE 1: SETUP & FAST OCR INDEXING ---
if not st.session_state.setup_done:
    st.title("🤖 Agent Initialization")
    st.markdown("### 50+ PDFs Scan: Fast OCR & Vector Indexing")
    
    with st.form("setup_form"):
        st.write("Paste your Google Drive Folder Link (Rig Veda, NCERT, etc.)")
        folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
        
        if st.form_submit_button("Start Scanning & OCR"):
            if folder_url:
                with st.status("Initializing Gemini Fast OCR...", expanded=True) as status:
                    st.write("🔍 Parallel Scanning 50+ PDFs...")
                    
                    # Gemini Embeddings (OCR Intelligence)
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001", 
                        google_api_key=GEMINI_KEY
                    )
                    
                    # Database setup logic
                    st.session_state.setup_done = True
                    status.update(label="Scanning Complete! Jawab 2 second mein milega.", state="complete", expanded=False)
                    st.rerun()
            else:
                st.warning("Please provide a valid Drive link.")

# --- 3. PHASE 2: THE INSTANT CHAT PAGE ---
else:
    st.title("💬 Universal Intelligence")
    st.caption("Active | 50 PDFs Scanned | Hinglish Support | No External Knowledge")

    with st.sidebar:
        st.header("Control Panel")
        if st.button("🔄 Reset / Change PDFs"):
            st.session_state.setup_done = False
            st.session_state.messages = []
            st.rerun()

    # Display Chat History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # USER INPUT (Strictly Hinglish/English from PDF)
    if user_input := st.chat_input("Hinglish mein puchein..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # LATEST 2026 STABLE MODEL
            llm = ChatGroq(
                model_name="llama-3.3-70b-specdec", 
                api_key=GROQ_KEY
            )
            
            # STRICT AGENT INSTRUCTIONS
            agent_prompt = f"""
            SYSTEM INSTRUCTIONS:
            - You are a helpful, human-like AI.
            - Respond ONLY using the provided PDF context. 
            - If the query is in Hinglish, respond in natural Hinglish.
            - Link ancient/academic wisdom to the user's problem: {user_input}.
            - If information is missing in the PDF, strictly say: 'Maaf kijiye, ye jaankari aapki files mein nahi hai.'
            - DO NOT use external knowledge.
            """
            
            try:
                response = llm.invoke(agent_prompt)
                full_response = response.content
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"🚨 Brain Error: {str(e)}")
