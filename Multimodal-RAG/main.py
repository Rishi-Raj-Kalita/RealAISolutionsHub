import os
import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_cloud_services import LlamaParse
from llama_index.core import Document

load_dotenv()

nest_asyncio.apply()


def get_model(model: str = 'llama3.1', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0)
        return llm
    elif (provider == 'aws'):
        from langchain_aws import ChatBedrockConverse
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        llm = ChatBedrockConverse(client=bedrock_client,
                                  model=model,
                                  temperature=0)
        return llm


def get_embeddings(model: str = 'llama3.1', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=model)
        return embeddings
    elif (provider == 'aws'):
        from langchain_aws import BedrockEmbeddings
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        embeddings = BedrockEmbeddings(client=bedrock_client, model_id=model)
        return embeddings


Settings.llm = get_model(provider='local', model='llama3.1')
Settings.embed_model = get_embeddings()


def parse_pages(pdf_path: str):
    parser = LlamaParse(api_key=os.getenv("LAMA_PARSE_API_KEY"),
                        parse_mode="parse_page_with_layout_agent")
    parsed_pdf = parser.get_json_result(pdf_path)
    pages = parsed_pdf[0]['pages']
    return pages


def index_document(pages):

    documents = []

    for i, page in enumerate(pages):
        # loop trough items of the page
        for item in page["items"]:
            document = Document(text=item["md"],
                                extra_info={
                                    "bbox": item["bBox"],
                                    "page": i
                                })
            documents.append(document)

    index = VectorStoreIndex.from_documents(documents)
    return index
