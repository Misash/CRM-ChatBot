# Load environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT models

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
     

chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)

messages = [
    SystemMessage(content="You are an expert data scientist"), # context
    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
]

response=chat(messages)

print(response.content,end='\n')