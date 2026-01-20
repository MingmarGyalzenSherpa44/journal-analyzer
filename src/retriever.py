from src.vector_store import VectorStoreService
from src.config import Config

class RetrieverService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": Config.TOP_K_RESULTS}
        )
    
    def get_retriever(self):
        return self.retriever
    
    def retrieve_documents(self, query):
        """Retrieve relevant documents for a query"""
        return self.retriever.get_relevant_documents(query)