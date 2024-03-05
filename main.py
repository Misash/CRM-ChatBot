# Load environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

#defining LLm model
llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")

 
from langchain.prompts.chat import ChatPromptTemplate


# template for CRM sales assistant
crm_template = (
    """ You are an expert sales assistant specialized in WhatsApp interactions. 
    You understand the nuances of customer communication and are skilled in providing 
    product information, answering queries, and guiding customers through the purchasing 
    process. Your advice is based on the latest sales strategies and CRM best practices."""
)

human_template = (
    """ Customer Inquiry: {customer_inquiry} Provide a detailed response that guides the 
    customer through their query, offers relevant product information, and encourages a 
    "positive sales outcome."""
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", crm_template),
    ("human", human_template),
])

# llm chain
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=chat_prompt)


# response 
response = chain.invoke({
    "customer_inquiry": "Can you tell me more about the features of your latest product?"
})

print(response['text'])



