import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from src.config import Config
from src.embeddings import EmbeddingService

class VectorStoreService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embeddings = self.embedding_service.get_embeddings()
        self.vector_db_path = Config.VECTOR_DB_PATH
        
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_journals(self, journals_dir="./data/journals"):
        """Load journal entries from directory"""
        loader = DirectoryLoader(
            journals_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        return documents
    
    def create_vector_store(self, documents):
        """Create vector store from documents"""
        texts = self.text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path,
            collection_name="journal_entries"
        )
        
        return vectorstore
    
    def load_vector_store(self):
        """Load existing vector store"""
        vectorstore = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings,
            collection_name="journal_entries"
        )
        return vectorstore