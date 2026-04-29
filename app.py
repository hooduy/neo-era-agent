import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="NEO-ERA Multilingual Agent", layout="centered", page_icon="🤖")

try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🔑 Secrets missing! Streamlit Settings mein check karein.")
    st.stop()

if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. PHASE 1: SETUP ---
if not st.session_state.setup_done:
    st.title("🤖 Universal Language Agent")
    with st.form("setup_form"):
        st.info("Scan your Drive folder for Multilingual Support (Odia/Hindi/English).")
        folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
        if st.form_submit_button("Initialize & Scan"):
            if folder_url:
                st.session_state.setup_done = True
                st.rerun()

# --- 3. PHASE 2: THE MULTILINGUAL CHAT ---
else:
    st.title("💬 Multilingual Intelligence")
    st.caption("Active | Supports: Odia, Hindi, English, Hinglish, Odianglish")

    with st.sidebar:
        if st.button("🔄 Reset Session"):
            st.session_state.setup_done = False
            st.session_state.messages = []
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if user_input := st.chat_input("Write in your language..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # LATEST STABLE MODEL
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_KEY, temperature=0.6)
            
            # --- THE MULTILINGUAL "SOUL" PROMPT ---
            multilingual_prompt = f"""
            You are a professional, empathetic multilingual mentor.
            The user query is: '{user_input}'
            
            STRICT LANGUAGE RULES:
            1. Language Detection: Detect the language of the user query.
            2. Language Matching: Respond ONLY in the same language/dialect as the user.
               - English query -> Professional English.
               - Hindi query -> Clean Hindi.
               - Hinglish query -> Natural Hinglish.
               - Odia query -> Proper Odia.
               - Odianglish query -> Friendly Odianglish (mix of Odia/English).
            
            INSTRUCTIONS:
            - Use ONLY the provided PDF context for facts.
            - Explain the concepts deeply, connecting them to the user's life.
            - If info is missing in PDF, say it politely in the SAME language.
            - Do not be a robotic translator; be a wise mentor.
            """
            
            try:
                with st.spinner("Processing in your language..."):
