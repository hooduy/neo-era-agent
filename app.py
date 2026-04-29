import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Universal AI Agent", layout="centered", page_icon="🤖")

# --- 2. SECRETS LOADING ---
try:
    # We use .get to prevent the app from crashing if a key is missing
    GROQ_KEY = st.secrets.get("GROQ_API_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
    
    if not GROQ_KEY or not GEMINI_KEY:
        st.error("🔑 Error: API Keys not found in Secrets. Please check your Streamlit Settings.")
        st.stop()
except Exception as e:
    st.error(f"Secrets Error: {e}")
    st.stop()

# Initialize session states
if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. POPUP (FOLDER LINK) ---
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

# --- 4. HUMAN-LIKE CHAT (HINDI & ENGLISH SUPPORT) ---
else:
    st.title("💬 Universal Intelligence")
    
    with st.sidebar:
        if st.button("➕ Start New Chat"):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()

    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # USER INPUT
    if user_input := st.chat_input("Ask anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                # The Brain: Using Llama 3 70B for maximum speed and empathy
                llm = ChatGroq(
                    model_name="llama3-70b-8192", 
                    api_key=GROQ_KEY,
                    temperature=0.6 # Slightly higher for more 'human' feeling
                )
                
                # Professional Instruction for Hindi/Hinglish support
                system_instruction = f"""
                You are a helpful, empathetic human-like AI assistant. 
                1. If the user speaks in Hindi or Hinglish (like 'man sahi nahi lag raha'), 
                   respond with care and support in that same language.
                2. Be concise but warm.
                3. If relevant, use info from their documents.
                
                User: {user_input}
                """
                
                response = llm.invoke(system_instruction)
                answer = response.content
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                if "401" in str(e):
                    st.error("❌ Invalid API Key! Please check your Groq key in Secrets.")
                else:
                    st.error(f"⚠️ Brain Error: {e}")
