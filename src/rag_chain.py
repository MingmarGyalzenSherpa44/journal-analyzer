from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.llm import LLMService
from src.retriever import RetrieverService

class RAGChain:
    def __init__(self, retriever):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        self.retriever = retriever
        
        self.prompt_template = """You are an AI assistant analyzing personal journal entries. 
        Use the following pieces of context from the journal entries to answer the question. 
        If you don't know the answer based on the context, say so. Don't make up information.
        Be empathetic and thoughtful in your responses.

        Context: {context}

        Question: {question}

        Answer: """
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def create_chain(self):
        """Create RAG chain"""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.PROMPT}
        )
        return qa_chain
    
    def query(self, question):
        """Query the RAG chain"""
        chain = self.create_chain()
        response = chain.invoke({"query": question})
        return response