import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

def apply_custom_styles():
    """Apply custom CSS styles to Streamlit app."""
    st.markdown("""
    <style>
        .main { background-color: #1a1a1a; color: #ffffff; }
        .sidebar .sidebar-content { background-color: #2d2d2d; }
        .stTextInput textarea { color: #ffffff !important; }
        .stSelectbox div[data-baseweb="select"] { color: white !important; background-color: #3d3d3d !important; }
        .stSelectbox svg { fill: white !important; }
        .stSelectbox option { background-color: #2d2d2d !important; color: white !important; }
        div[role="listbox"] div { background-color: #2d2d2d !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """Configure the sidebar for model selection and display capabilities."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
        st.divider()
        st.markdown("### Model Capabilities")
        st.markdown("""
        - 🐍 Python Expert
        - 🐞 Debugging Assistant
        - 📝 Code Documentation
        - 💡 Solution Design
        """)
        st.divider()
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    return selected_model

def initialize_session():
    """Initialize session state variables."""
    if "message_log" not in st.session_state:
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

def display_chat():
    """Display the chat messages stored in session state."""
    with st.container():
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def generate_ai_response(prompt_chain, llm_engine):
    """Generate AI response using the prompt chain and language model."""
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    """Build the chat prompt chain from the message log."""
    prompt_sequence = [SystemMessagePromptTemplate.from_template(
        "You are an expert AI coding assistant. Provide concise, correct solutions "
        "with strategic print statements for debugging. Always respond in English."
    )]
    
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    
    return ChatPromptTemplate.from_messages(prompt_sequence)

def main():
    """Main function to run the Streamlit app."""
    st.title("🧠 DeepSeek Code Companion")
    st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")
    
    apply_custom_styles()
    selected_model = setup_sidebar()
    initialize_session()
    
    llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)
    display_chat()
    
    user_query = st.chat_input("Type your coding question here...")
    
    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        with st.spinner("🧠 Processing..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain, llm_engine)
        
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.rerun()

if __name__ == "__main__":
    main()
