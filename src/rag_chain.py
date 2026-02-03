from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.llm import LLMService

class RAGChain:
    def __init__(self, retriever):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        self.retriever = retriever
        
        # Use ChatPromptTemplate for chat models
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant analyzing personal journal entries. 
Use the following pieces of context from the journal entries to answer the question. 
If you don't know the answer based on the context, say so. Don't make up information.
Be empathetic and thoughtful in your responses.

Context: {context}"""),
            ("human", "{input}")
        ])
    
    def format_docs(self, docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self):
        """Create RAG chain using LCEL"""
        # Build the chain using LangChain Expression Language
        chain = (
            {
                "context": self.retriever | self.format_docs,
                "input": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def query(self, question):
        """Query the RAG chain"""
        chain = self.create_chain()
        
        # Get the answer
        answer = chain.invoke(question)
        
        source_docs = self.retriever.invoke(question) 
        
        return {
            "answer": answer,
            "context": source_docs,
            "input": question
        }