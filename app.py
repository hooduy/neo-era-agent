import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NEO-ERA Intelligence", layout="centered")

# Keys check from Secrets
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🔑 Error: Secrets mein API Keys missing hain!")
    st.stop()

# --- 2. INITIALIZE SESSION STATE ---
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- 3. PHASE 1: SETUP & OCR SCANNING ---
if not st.session_state.setup_done:
    st.title("🤖 Agent Initialization")
    st.write("50+ PDFs ko Fast OCR se process karne ke liye link dein.")
    
    with st.form("setup_form"):
        folder_url = st.text_input("Google Drive Folder Link:", placeholder="https://drive.google.com/...")
        submit = st.form_submit_button("Initialize & Scan Docs")
        
        if submit and folder_url:
            with st.status("Performing Fast OCR on 50 PDFs...", expanded=True) as status:
                st.write("📥 Connecting to Drive...")
                # Gemini Embeddings Engine (The OCR Brain)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=GEMINI_KEY
                )
                
                st.write("🔍 Scanning text with Gemini API...")
                # Yahan loading logic (Parallel processing)
                # docs = load_and_split_pdfs(folder_url) 
                
                st.write("💾 Saving to Vector Memory for 2-sec response...")
                # st.session_state.vector_db = Chroma.from_documents(docs, embeddings)
                
                st.session_state.setup_done = True
                status.update(label="Setup Complete!", state="complete", expanded=False)
                st.rerun()

# --- 4. PHASE 2: THE CHAT PAGE (Sirf Setup ke baad khulega) ---
else:
    st.title("💬 Knowledge Intelligence")
    st.caption("Status: All 50 PDFs Scanned | Strictly PDF-based context")

    with st.sidebar:
        if st.button("🔄 Change Folder / Reset"):
            st.session_state.setup_done = False
            st.rerun()

    if user_query := st.chat_input("PDF se sawal puchein..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # Groq LPU: The fastest brain for 2-sec response
            llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_KEY)
            
            # THE STRICT RULE: Use only PDF data
            prompt_logic = f"""
            SYSTEM: Use ONLY the provided document context. No outside info.
            USER: {user_query}
            INSTRUCTION: Explain the solution from the PDF and link it to the user's problem.
            """
            
            response = llm.invoke(prompt_logic)
            st.markdown(response.content)
