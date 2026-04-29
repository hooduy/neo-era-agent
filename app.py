import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(page_title="NEO-ERA Multilingual Agent", layout="centered", page_icon="🤖")

try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🔑 Secrets missing! Settings mein GEMINI_API_KEY aur GROQ_API_KEY check karein.")
    st.stop()

if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. PHASE 1: SETUP ---
if not st.session_state.setup_done:
    st.title("🤖 Universal Language Agent")
    with st.form("setup_form"):
        st.info("Drive folder scan karke Odia/Hindi/English support shuru karein.")
        folder_url = st.text_input("Folder Link:", placeholder="https://drive.google.com/...")
        if st.form_submit_button("Initialize & Scan"):
            if folder_url:
                st.session_state.setup_done = True
                st.rerun()

# --- 3. PHASE 2: MULTILINGUAL CHAT ---
else:
    st.title("💬 Multilingual Intelligence")
    st.caption("Supports: English, Hindi, Hinglish, Odia, Odianglish")

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
            # LATEST STABLE MODEL (Updated to avoid decommissioned error)
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_KEY, temperature=0.6)
            
            multilingual_prompt = f"""
            You are a wise mentor. The user says: '{user_input}'
            
            RULES:
            1. Respond ONLY in the same language as the user (English, Hindi, Odia, Hinglish, or Odianglish).
            2. Use the PDF context to explain shlokas or facts deeply.
            3. Link the knowledge to the user's real-life problem naturally.
            4. If the info isn't in the PDF, say it politely in the SAME language.
            """
            
            try:
                # Proper indentation here to fix your IndentationError
                with st.spinner("Processing..."):
                    response = llm.invoke(multilingual_prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"🚨 Brain Error: {str(e)}")
