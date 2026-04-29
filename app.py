import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. UI SETUP ---
st.set_page_config(page_title="Universal AI Agent", layout="centered")

# --- 2. AUTHENTICATION ---
try:
    # Pulling from the secrets you just fixed
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🔑 Secrets Error: Check your Streamlit Secrets formatting!")
    st.stop()

if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. POPUP (GOOGLE DRIVE LINK) ---
if not st.session_state.ready:
    st.title("🤖 Personal AI Assistant")
    with st.form("setup_form"):
        st.write("Paste your Google Drive Folder Link to begin.")
        folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
        if st.form_submit_button("Initialize Chat"):
            if folder_url:
                # Fast OCR and Initialization would happen here
                st.session_state.ready = True
                st.rerun()
            else:
                st.warning("Please provide a link.")

# --- 4. HUMAN-LIKE CHAT (HINDI/ENGLISH) ---
else:
    st.title("💬 Universal Intelligence")
    
    if st.sidebar.button("➕ New Session"):
        st.session_state.ready = False
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # The Brain: Groq Llama 3 (Hyper-Fast)
            llm = ChatGroq(
                model_name="llama3-70b-8192", 
                api_key=GROQ_KEY, # Using the fixed key
                temperature=0.5
            )
            
            # Instruction for human-like response and Hindi support
            prompt = f"""
            You are a helpful human-like AI. 
            If the user speaks in Hindi/Hinglish, reply in Hindi/Hinglish.
            Be empathetic and fast.
            User Query: {user_input}
            """
            
            # Execute and Display
            response = llm.invoke(prompt)
            answer = response.content
            
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
