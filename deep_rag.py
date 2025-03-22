import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Constants
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Updated Prompt for Detailed, Readable Answers
PROMPT_TEMPLATE = """
You are a highly knowledgeable research assistant. Provide a **detailed, structured, and well-formatted** answer.
Use bullet points, paragraphs, and numbered lists where appropriate.
Always **include key details and insights** from the document.

**Query:** {user_query}
**Context:** {document_context}

### Answer:
"""

# Custom Styling
st.markdown("""
    <style>
    body { background-color: #121212; color: #EAEAEA; font-family: 'Arial', sans-serif; }
    .stApp { background-color: #1E1E1E; padding: 20px; border-radius: 10px; }
    .stTitle, .stMarkdown { color: #00FFA3; text-align: center; font-weight: bold; }
    .stFileUploader { border-radius: 8px; padding: 15px; background-color: #292929; }
    .stChatInput input { background-color: #292929; color: #EAEAEA; border: 1px solid #3A3A3A; }
    .stButton button { background-color: #00FFA3; color: #121212; border-radius: 6px; font-weight: bold; }
    .stChatMessage { border-radius: 8px; padding: 12px; margin: 8px 0; }
    </style>
    """, unsafe_allow_html=True)

# Functions
def save_uploaded_file(uploaded_file):
    """Save uploaded file and return its path."""
    file_path = uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def process_document(file_path):
    """Extract and split text into meaningful chunks for embedding."""
    raw_docs = PDFPlumberLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(raw_docs)
    DOCUMENT_VECTOR_DB.add_documents(chunks)

def get_answer(user_query):
    """Retrieve relevant document sections and generate a structured answer."""
    related_docs = DOCUMENT_VECTOR_DB.similarity_search(user_query)
    
    if not related_docs:
        return "**No relevant information found. Try rephrasing your question.**"
    
    # Format context text for readability
    context_text = "\n\n".join([f"📄 **Page {doc.metadata.get('page', 'Unknown')}**:\n{doc.page_content}" for doc in related_docs])
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response = (prompt | LANGUAGE_MODEL).invoke({"user_query": user_query, "document_context": context_text})

    return response

# UI Setup
st.title("📘 IntelliDoc AI")
st.markdown("### Your Smart Document Assistant")
st.markdown("---")

# File Upload
document = st.file_uploader("📂 Upload a Research PDF", type="pdf", help="Select a PDF for AI analysis")
if document:
    process_document(save_uploaded_file(document))
    st.success("✅ Document processed! Ask your question below.")
    
    user_query = st.chat_input("🔍 Enter your question...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("🔎 Analyzing..."):
            response = get_answer(user_query)
        
        # Display Response with Readability
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)



