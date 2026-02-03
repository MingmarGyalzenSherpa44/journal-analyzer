import boto3
from langchain_aws import ChatBedrock
from src.config import Config

class LLMService:
    def __init__(self):
        session_kwargs = {"region_name": Config.AWS_REGION}
        if Config.AWS_ACCESS_KEY_ID:
            session_kwargs.update({
                "aws_access_key_id": Config.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": Config.AWS_SECRET_ACCESS_KEY
            })
        if Config.AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = Config.AWS_SESSION_TOKEN

        self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)

        # Use ChatBedrock for chat/conversational models like Amazon Nova
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id=Config.LLM_MODEL_ID,  # amazon.nova-lite-v1:0
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 2048
            }
        )

    def get_llm(self):
        return self.llm