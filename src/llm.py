import boto3
from langchain_aws import ChatBedrock
from src.config import Config

class LLMService:
    def __init__(self):
        # Build session kwargs dynamically
        session_kwargs = {
            'region_name': Config.AWS_REGION
        }
        
        # Add credentials if provided
        if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            session_kwargs['aws_access_key_id'] = Config.AWS_ACCESS_KEY_ID
            session_kwargs['aws_secret_access_key'] = Config.AWS_SECRET_ACCESS_KEY
        
        # Add session token if provided (for temporary credentials)
        if Config.AWS_SESSION_TOKEN:
            session_kwargs['aws_session_token'] = Config.AWS_SESSION_TOKEN
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            **session_kwargs
        )
        
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id=Config.LLM_MODEL_ID,
            model_kwargs={
                "max_tokens": 2048,
                "temperature": 0.7,
            }
        )
    
    def get_llm(self):
        return self.llm