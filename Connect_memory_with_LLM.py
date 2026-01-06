import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler


# ============================================
# Configuration & Logging Setup
# ============================================
class Config:
    """Centralized configuration management"""
    MODEL_NAME = "llama-3.1-8b-instant"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "vectorstore/db_faiss"
    TOP_K_RESULTS = 3
    LOG_FILE = "logs/rag_assistant.log"

def setup_logging():
    """Configure logging with file and console handlers"""
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
# Custom Callback Handler for Streaming
# ============================================
class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback for handling streaming responses"""
    
    def on_llm_start(self, *args, **kwargs):
        logger.info("🤖 Assistant is thinking...")
    
    def on_llm_end(self, *args, **kwargs):
        logger.info("✓ Response generated successfully")

# ============================================
# RAG Healthcare Assistant Class
# ============================================
class HealthcareRAGAssistant:
    """Professional RAG-based Healthcare Assistant"""
    
    def __init__(self):
        """Initialize the RAG assistant with all components"""
        logger.info("Initializing Healthcare RAG Assistant...")
        
        load_dotenv()
        self._validate_environment()
        
        self.llm = self._initialize_llm()
        self.retriever = self._initialize_retriever()
        self.rag_chain = self._build_rag_chain()
        
        logger.info("✓ Healthcare RAG Assistant initialized successfully")
    
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = ["GROQ_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the language model"""
        try:
            logger.info(f"Loading LLM: {Config.MODEL_NAME}")
            llm = ChatGroq(
                model=Config.MODEL_NAME,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                callbacks=[StreamingCallbackHandler()]
            )
            logger.info("✓ LLM loaded successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_retriever(self):
        """Initialize embeddings and vector store retriever"""
        try:
            logger.info(f"Loading embeddings: {Config.EMBEDDING_MODEL}")
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL
            )
            
            logger.info(f"Loading vector database from: {Config.VECTOR_DB_PATH}")
            if not os.path.exists(Config.VECTOR_DB_PATH):
                raise FileNotFoundError(
                    f"Vector database not found at: {Config.VECTOR_DB_PATH}\n"
                    "Please ensure the database is created before running."
                )
            
            db = FAISS.load_local(
                Config.VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            retriever = db.as_retriever(
                search_kwargs={"k": Config.TOP_K_RESULTS}
            )
            
            logger.info("✓ Vector database and retriever loaded successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise
    
    def _build_rag_chain(self):
        """Build the RAG chain pipeline"""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional healthcare assistant with expertise in medical information. "
             "Your role is to provide accurate, helpful, and empathetic responses based on the provided context.\n\n"
             "Guidelines:\n"
             "- Use the context below to answer questions accurately\n"
             "- If the answer is not in the context, clearly state that you don't have that information\n"
             "- Provide clear, concise explanations that are easy to understand\n"
             "- Always recommend consulting healthcare professionals for medical decisions\n"
             "- Use a warm, professional, and supportive tone\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])
        
        rag_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
        )
        
        logger.info("✓ RAG chain built successfully")
        return rag_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return structured response
        
        Args:
            question: User's healthcare question
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Please provide a valid question",
                "answer": None
            }
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Get relevant documents
            docs = self.retriever.invoke(question)

            
            # Invoke RAG chain
            response = self.rag_chain.invoke(question)
            
            result = {
                "success": True,
                "answer": response.content,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ],
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(docs)
            }
            
            logger.info("✓ Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": None
            }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format the response for display"""
        if not result["success"]:
            return f"\n❌ Error: {result['error']}\n"
        
        output = [
            "\n" + "=" * 80,
            "🏥 HEALTHCARE ASSISTANT RESPONSE",
            "=" * 80,
            "",
            result["answer"],
            "",
            "─" * 80,
            f"📚 Sources Referenced: {result['num_sources']}",
            f"⏰ Response Time: {result['timestamp']}",
            "─" * 80,
            "",
            "⚠️  DISCLAIMER: This information is for educational purposes only.",
            "Please consult with qualified healthcare professionals for medical advice.",
            "=" * 80,
            ""
        ]
        
        return "\n".join(output)

# ============================================
# Interactive CLI Interface
# ============================================
def print_banner():
    """Display welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║         🏥 HEALTHCARE RAG ASSISTANT v2.0                         ║
║         Professional Medical Information Retrieval System         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

Welcome! Ask me any healthcare-related question.
Type 'exit', 'quit', or 'q' to end the session.
Type 'help' for usage instructions.
"""
    print(banner)

def print_help():
    """Display help information"""
    help_text = """
📖 USAGE INSTRUCTIONS:
─────────────────────────────────────────────────────────────────

• Ask any healthcare-related question
• Be specific for better results
• Type 'exit', 'quit', or 'q' to end session
• Type 'clear' to clear the screen
• Type 'help' to see this message again

Example questions:
  - What are the symptoms of diabetes?
  - How can I manage high blood pressure?
  - What should I know about COVID-19 vaccines?

─────────────────────────────────────────────────────────────────
"""
    print(help_text)

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main application entry point"""
    try:
        print_banner()
        
        # Initialize assistant
        assistant = HealthcareRAGAssistant()
        
        print("\n✓ System ready! You can start asking questions.\n")
        
        # Interactive loop
        while True:
            try:
                query = input("🔍 Your Question: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using Healthcare RAG Assistant. Stay healthy!\n")
                    break
                elif query.lower() == 'help':
                    print_help()
                    continue
                elif query.lower() == 'clear':
                    clear_screen()
                    print_banner()
                    continue
                
                # Process query
                result = assistant.query(query)
                formatted_response = assistant.format_response(result)
                print(formatted_response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!\n")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {str(e)}")
                print(f"\n❌ An unexpected error occurred: {str(e)}\n")
                
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        print(f"\n❌ Critical error initializing system: {str(e)}\n")
        sys.exit(1)

# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    main()