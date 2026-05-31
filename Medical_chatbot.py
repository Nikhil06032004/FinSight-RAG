import os
import sys
import logging
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Configuration
# ============================================
class Config:
    """Centralized configuration"""
    MODEL_NAME = "llama-3.1-8b-instant"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DB_FAISS_PATH = "vectorstore/db_faiss"
    TOP_K_RESULTS = 5
    LOG_FILE = "logs/streamlit_rag.log"

# ============================================
# Logging Setup
# ============================================
def setup_logging():
    """Configure logging"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================
# Custom CSS Styling
# ============================================
def load_custom_css():
    """Load custom CSS for better UI"""
    st.markdown("""
        <style>
        .main { padding: 2rem; }

        .header-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1a1a2e;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .stMarkdown strong { color: #0f3460; }

        .stMarkdown ul, .stMarkdown ol {
            margin-left: 1.5rem;
            line-height: 1.8;
        }

        .stMarkdown p {
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        details {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-radius: 5px;
        }

        details summary {
            font-weight: 600;
            cursor: pointer;
            padding: 0.5rem;
            color: #0f3460;
        }

        details summary:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }

        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================
# Initialize Components
# ============================================
@st.cache_resource(show_spinner=False)
def initialize_embeddings():
    """Initialize and cache embeddings model"""
    try:
        logger.info(f"Loading embeddings: {Config.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        logger.info("✓ Embeddings loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings: {str(e)}")
        st.error(f"Error loading embeddings: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def initialize_vectorstore(_embeddings):
    """Initialize and cache vector store"""
    try:
        logger.info(f"Loading vector database from: {Config.DB_FAISS_PATH}")

        if not os.path.exists(Config.DB_FAISS_PATH):
            error_msg = f"Vector database not found at: {Config.DB_FAISS_PATH}"
            logger.error(error_msg)
            st.error(error_msg)
            return None

        db = FAISS.load_local(
            Config.DB_FAISS_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✓ Vector database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        st.error(f"Error loading vector database: {str(e)}")
        return None

def initialize_llm():
    """Initialize LLM with API key from environment or session state"""
    try:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key")

        if not api_key:
            return None

        logger.info(f"Initializing LLM: {Config.MODEL_NAME}")
        llm = ChatGroq(
            model=Config.MODEL_NAME,
            api_key=api_key,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        logger.info("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def create_rag_chain(llm, retriever):
    """Create the RAG chain pipeline"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional financial regulatory compliance assistant for BNP Paribas, "
             "specialized in credit risk, banking regulations, and financial reporting.\n\n"
             "Guidelines:\n"
             "- Answer STRICTLY from the provided context only — no external knowledge\n"
             "- Cite the source document for every key point in your response\n"
             "- Address EU, APAC, and GLOBAL regions separately when applicable\n"
             "- Explain financial and regulatory terms in simple language when first mentioned\n"
             "- Include relevant regulatory thresholds, ratios, and compliance requirements\n"
             "- Highlight risk implications and compliance obligations where applicable\n"
             "- If context is insufficient, state: 'This is not covered in the available regulatory documents'\n"
             "- Never provide speculative financial advice or investment recommendations\n"
             "- Maintain a concise, precise, and professional banking tone throughout\n\n"
             "Format your response in a structured, professional way:\n"
             "- Use **bold** for key regulatory terms, ratios, and critical thresholds\n"
             "- Use clear sections with descriptive headings\n"
             "- Use bullet points for lists, requirements, and multiple items\n"
             "- Always end with the source document reference\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("✓ RAG chain created successfully")
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

# ============================================
# Helper Functions
# ============================================
def format_source_documents(docs: List) -> str:
    """Format source documents for display"""
    if not docs:
        return ""

    formatted = "\n\n---\n\n"
    formatted += '<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #0f3460; margin-top: 1.5rem;">'
    formatted += f'<strong style="color: #0f3460;">📚 Sources Referenced:</strong> {len(docs)} document(s) | '
    formatted += '<em style="color: #6c757d; font-size: 0.9rem;">Click to expand for source details</em>'
    formatted += '</div>'

    formatted += '\n\n<details style="margin-top: 0.5rem;">'
    formatted += '<summary style="cursor: pointer; color: #0f3460; font-weight: 600; padding: 0.5rem;">View Source Details</summary>'
    formatted += '<div style="padding: 1rem; background-color: #f8f9fa; border-radius: 5px; margin-top: 0.5rem;">'

    for i, doc in enumerate(docs, 1):
        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content

        source_info = "Unknown"
        if doc.metadata:
            if 'source' in doc.metadata:
                source_info = doc.metadata['source']
            elif 'file' in doc.metadata:
                source_info = doc.metadata['file']

        formatted += f"""
<div style="margin-bottom: 0.8rem; padding: 0.5rem; background-color: white; border-radius: 5px; border-left: 3px solid #0f3460;">
    <strong style="color: #0f3460;">Source {i}:</strong> <em style="color: #6c757d; font-size: 0.85rem;">{source_info}</em><br>
    <span style="color: #495057; font-size: 0.9rem;">{content_preview}</span>
</div>
"""

    formatted += '</div></details>'
    return formatted

def display_header():
    """Display app header"""
    st.markdown("""
        <div class="header-container">
            <div class="header-title">💹 Finance RAG Assistant</div>
            <div class="header-subtitle">AI-Powered Financial Regulatory Information Retrieval System</div>
        </div>
    """, unsafe_allow_html=True)

def display_disclaimer():
    """Display financial disclaimer"""
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Financial Disclaimer:</strong><br>
            This information is for reference and educational purposes only and should not be considered
            as financial or investment advice. Always consult qualified financial and compliance professionals
            for regulatory decisions.
        </div>
    """, unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================
def render_sidebar(embeddings, vectorstore):
    """Render sidebar with settings and information"""
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("🔑 API Configuration")
        env_api_key = os.getenv("GROQ_API_KEY")

        if env_api_key:
            st.success("✓ API Key loaded from environment")
            st.session_state.groq_api_key = env_api_key
        else:
            api_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                value=st.session_state.get("groq_api_key", ""),
                help="Get your API key from https://console.groq.com"
            )
            if api_key:
                st.session_state.groq_api_key = api_key
                st.success("✓ API Key configured")

        st.divider()

        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Embeddings", "✓ Ready" if embeddings else "✗ Failed")
        with col2:
            st.metric("Vector DB", "✓ Ready" if vectorstore else "✗ Failed")

        if vectorstore:
            st.metric("Documents", f"{vectorstore.index.ntotal:,}")

        st.divider()

        st.subheader("🤖 Model Settings")
        st.info(f"""
        **Model:** {Config.MODEL_NAME}  
        **Temperature:** {Config.TEMPERATURE}  
        **Max Tokens:** {Config.MAX_TOKENS}  
        **Top K Results:** {Config.TOP_K_RESULTS}
        """)

        st.divider()

        st.subheader("📚 Loaded Documents")
        st.markdown("""
        - 🌍 Basel III Framework
        - 🌍 IFRS 9 Standard
        - 🌏 RBI Credit Risk Guidelines
        - 🇪🇺 ECB Banking Supervision Guide
        - 🌍 BNP Paribas Annual Report 2023
        """)

        st.divider()

        st.subheader("🔄 Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("📥 Export Chat", use_container_width=True):
            if "messages" in st.session_state and st.session_state.messages:
                chat_export = "\n\n".join([
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.messages
                ])
                st.download_button(
                    "Download Chat History",
                    chat_export,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        st.divider()

        st.markdown("""
            <div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
                <p>Finance RAG Assistant v1.0</p>
                <p>Powered by LangChain & Groq</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================
# Main Application
# ============================================
def main():
    """Main application function"""

    load_custom_css()
    display_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("🔄 Initializing system components..."):
        embeddings = initialize_embeddings()
        vectorstore = initialize_vectorstore(embeddings) if embeddings else None

    render_sidebar(embeddings, vectorstore)

    if not embeddings or not vectorstore:
        st.error("❌ System initialization failed. Please check the logs and ensure all requirements are met.")
        st.stop()

    if not st.session_state.get("groq_api_key"):
        st.warning("⚠️ Please configure your Groq API key in the sidebar to continue.")
        st.info("""
        **How to get started:**
        1. Get a free API key from [Groq Console](https://console.groq.com)
        2. Enter the API key in the sidebar
        3. Start asking questions!
        """)
        st.stop()

    llm = initialize_llm()
    if not llm:
        st.error("❌ Failed to initialize LLM. Please check your API key.")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K_RESULTS})
    rag_chain = create_rag_chain(llm, retriever)

    if not rag_chain:
        st.error("❌ Failed to create RAG chain.")
        st.stop()

    display_disclaimer()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about credit risk, Basel norms, IFRS 9, RBI guidelines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving from regulatory documents..."):
                try:
                    docs = retriever.invoke(prompt)
                    response = rag_chain.invoke(prompt)

                    full_response = response
                    if docs:
                        full_response += format_source_documents(docs)

                    st.markdown(full_response, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                    logger.info(f"Successfully processed query: {prompt[:100]}...")

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error processing query: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Example Questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Credit Risk:**
        - What is the minimum CET1 capital ratio under Basel III?
        - How does IFRS 9 define credit impairment and expected credit loss?
        - What are the RBI guidelines on credit risk management?
        
        **Regulatory Compliance:**
        - What are the Liquidity Coverage Ratio requirements under Basel III?
        - How does the ECB supervise capital adequacy in European banks?
        - What are the disclosure requirements under IFRS 9?
        
        **Financial Reporting:**
        - What was BNP Paribas credit risk exposure in 2023?
        - How are risk-weighted assets calculated under Basel III?
        - What are the key differences between Tier 1 and Tier 2 capital?
        """)

    # Tips Section
    with st.expander("ℹ️ Tips for Better Responses", expanded=False):
        st.markdown("""
        **Get the most precise answers:**
        - Be specific about the regulation or framework you are referring to
        - Mention the region if relevant (e.g., "under RBI guidelines", "for EU banks")
        - Ask about thresholds, ratios, requirements, or compliance obligations
        - Ask follow-up questions for deeper regulatory context
        
        **Example of a good question:**
        - ❌ "Tell me about credit risk"
        - ✅ "What are the RBI guidelines on credit risk provisioning for APAC banks?"
        """)

# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        st.error(f"❌ Critical error: {str(e)}")
        st.error("Please check the logs for more information.")