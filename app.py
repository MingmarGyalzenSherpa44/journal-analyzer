import streamlit as st
from src.vector_store import VectorStoreService
from src.retriever import RetrieverService
from src.rag_chain import RAGChain
import os

st.set_page_config(
    page_title="Journal Analyzer",
    page_icon="📔",
    layout="wide"
)

st.title("📔 Journal Analyzer - RAG Application")
st.markdown("Ask questions about your journal entries using AI")

# Initialize session state
if 'vectorstore_created' not in st.session_state:
    st.session_state.vectorstore_created = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("Setup")
    
    if st.button("🔄 Load/Create Vector Store"):
        with st.spinner("Processing journal entries..."):
            try:
                vector_service = VectorStoreService()
                
                # Check if vector store exists
                if os.path.exists(vector_service.vector_db_path) and os.listdir(vector_service.vector_db_path):
                    vectorstore = vector_service.load_vector_store()
                    st.success("Loaded existing vector store!")
                else:
                    # Load and create new vector store
                    documents = vector_service.load_journals()
                    if not documents:
                        st.error("No journal entries found in ./data/journals/")
                    else:
                        vectorstore = vector_service.create_vector_store(documents)
                        st.success(f"Created vector store with {len(documents)} documents!")
                
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_created = True
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Vector Store"):
        import shutil
        from src.config import Config
        if os.path.exists(Config.VECTOR_DB_PATH):
            shutil.rmtree(Config.VECTOR_DB_PATH)
            st.session_state.vectorstore_created = False
            st.success("Vector store cleared!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses RAG (Retrieval Augmented Generation) to:
    - Index your journal entries
    - Retrieve relevant context
    - Generate insightful answers
    """)

# Main area
if st.session_state.vectorstore_created:
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander("View Sources"):
                    for i, doc in enumerate(chat["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
    
    # Query input
    question = st.chat_input("Ask a question about your journals...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever_service = RetrieverService(st.session_state.vectorstore)
                    retriever = retriever_service.get_retriever()
                    
                    rag_chain = RAGChain(retriever)
                    response = rag_chain.query(question)
                    
                    answer = response['result']
                    sources = response.get('source_documents', [])
                    
                    st.write(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Please load or create the vector store from the sidebar to begin.")
    st.markdown("""
    ### Getting Started:
    1. Add your journal entries (txt files) to `./data/journals/`
    2. Click 'Load/Create Vector Store' in the sidebar
    3. Start asking questions about your journals!
    
    ### Example Questions:
    - What were my main concerns last month?
    - How has my mood changed over time?
    - What activities made me happy?
    - Summarize my thoughts about work
    """)