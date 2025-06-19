import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
import boto3
import boto3
import asyncio
from lmnr import Laminar

load_dotenv()

access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')
laminar_api_key = os.getenv('LAMINAR')


def get_model(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):

        llm = ChatOllama(model=model, temperature=0.8)
        return llm
    elif (provider == 'llama'):

        llm = ChatOllama(model='llama3.1', temperature=0.8)
        return llm
    elif (provider == 'aws'):

        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        llm = ChatBedrockConverse(client=bedrock_client,
                                  model=model,
                                  temperature=0.8)
        return llm


bedrock_client = boto3.client('bedrock-runtime',
                              region_name='us-east-1',
                              aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key)

llm = get_model(model='anthropic.claude-3-5-sonnet-20240620-v1:0',
                provider='aws')

initial_actions = [{'open_tab': {'url': 'https://www.swiggy.com/'}}]

browser = Browser(config=BrowserConfig(
    chrome_instance_path=
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
))

tasks = """
1. Click on Search box of landing page.
2. Type "Harley’s Fine Baking" in the search box ("Search for restaurants and food") and click on the search icon.
3. Click on the first result for "Harley’s Fine Baking"
4. Look for "Search For Dishes" search box in the page and look for the dish "Chef's Favourite Blueberry Cheesecake With Macaron"
5. Select the first result for "Chef's Favourite Blueberry Cheesecake With Macaron" by clicking on Add button 
6. After clicking on Add button view cart option appears with a bag icon on the bottom section of the browser. Click on that icon.
7. Next you need to select the delivery location select the 4th option with delivery location Leha Residency 2nd Floor, click on Deliver here button.
8. Next, it Will ask you for proceed to pay, click on proceed to pay button.
9. Next, it will take you to payments page with Pay on delivery (Cash/UPI) as the preferred payment method, click on pay with cash button to confirm the order. 
10. Task completes.
"""

Laminar.initialize(project_api_key=laminar_api_key)


async def main():
    agent = Agent(
        task=tasks,
        initial_actions=initial_actions,
        llm=llm,
        save_conversation_path="logs/conversation",
        browser=browser,
        use_vision=True,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
