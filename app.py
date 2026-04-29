import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="NEO-ERA AI Agent", layout="centered")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🔑 API Keys Missing! Add them to Streamlit Cloud Settings.")
    st.stop()

if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. STARTUP INITIALIZATION POP-UP ---
if not st.session_state.ready:
    st.title("🤖 Project NEO-ERA: AI Setup")
    st.markdown("### Industrial Mining & Waste Management Intelligence")
    
    with st.container():
        with st.form("setup_form"):
            # Professional Drag-and-Drop instead of complex Drive Login
            uploaded_files = st.file_uploader("Upload Mining/Industrial PDFs:", type="pdf", accept_multiple_files=True)
            
            if st.form_submit_button("Initialize Agent"):
                if uploaded_files:
                    with st.spinner("Gemini 3 Flash performing Fast OCR..."):
                        # Save files locally for the agent to read
                        if not os.path.exists("temp_docs"):
                            os.makedirs("temp_docs")
                        
                        full_docs = []
                        for f in uploaded_files:
                            path = os.path.join("temp_docs", f.name)
                            with open(path, "wb") as buffer:
                                buffer.write(f.getvalue())
                            
                            loader = PyPDFLoader(path)
                            full_docs.extend(loader.load())

                        # Brain Setup
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        splits = text_splitter.split_documents(full_docs)
                        
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
                        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                        
                        st.session_state.retriever = vectorstore.as_retriever()
                        st.session_state.ready = True
                        st.rerun()
                else:
                    st.warning("Please upload at least one PDF to proceed.")

# --- 3. THE CHATGPT WORKSPACE ---
else:
    st.title("💬 NEO-ERA Intelligence")
    
    with st.sidebar:
        st.header("Control Panel")
        if st.button("➕ New Session"):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()

    # Display History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Human-like Chat Input
    if user_input := st.chat_input("Ask about NEO-ERA REGEN or your mining docs..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # AGENT LOGIC: Groq + Native Multilingual Translation
            llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0.1)
            
            # This 'Chain' looks at your PDFs and answers
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=st.session_state.retriever
            )
            
            # The Prompt that handles Hindi/English and Human-tone
            prompt_instruction = f"""
            You are the NEO-ERA Industrial Expert. 
            User Query: {user_input}
            If the query is in Hindi, respond in Hindi. 
            If the PDFs contain Hindi, translate accurately.
            Be helpful, professional, and explain technical mining terms simply.
            """
            
            response = qa_chain.run(prompt_instruction)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
