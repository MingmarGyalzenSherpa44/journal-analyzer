import boto3
from langchain_aws import BedrockEmbeddings
from src.config import Config

class EmbeddingService:
    def __init__(self):
        # Build session kwargs dynamically
        session_kwargs = {
            'region_name': Config.AWS_REGION
        }
        
        # Add credentials if provided
        if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            session_kwargs['aws_access_key_id'] = Config.AWS_ACCESS_KEY_ID
            session_kwargs['aws_secret_access_key'] = Config.AWS_SECRET_ACCESS_KEY
        
        if Config.AWS_SESSION_TOKEN:
            session_kwargs['aws_session_token'] = Config.AWS_SESSION_TOKEN
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            **session_kwargs
        )
        
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=Config.EMBEDDING_MODEL_ID
        )
    
    def get_embeddings(self):
        return self.embeddings