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
    """ Customer Inquiry: {customer_inquiry} Provide a concise response that guides the 
    customer through their query, offers relevant product information, and encourages a 
    positive sales outcome."""
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", crm_template),
    ("human", human_template),
])

# llm chain
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=chat_prompt)


# response 
# response = chain.invoke({
#     "customer_inquiry": "Can you tell me more about the features of your latest product?"
# })

# print(response['text'])

# Embeddings and Vector Stores
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# get the data on chunks
loader = TextLoader("./data/enterprise_data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# print chunks
# for i in range(len(docs)):
#     print(f"CHUNK {i}:\n\n", docs[i].page_content)


#turn the chunks on embeddings
embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query(docs[0].page_content)
# print(query_result)

from langchain_pinecone import PineconeVectorStore
index_name = "index"
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


# vector similarity search
# query = "What is the company mission?"
# result  = docsearch.similarity_search(query)
# print(result [0].page_content)


def answer_question(question):
    #find relevant content on vector emdeddings
    search_result = docsearch.similarity_search(question)
    relevant_content = search_result[0].page_content
    
    # concatenate revelant content with crm_template
    response_prompt = f"""
    {crm_template}
    
    Customer Inquiry: {question}
    
    Based on the following information: {relevant_content}
    """
    
    response = chain.invoke({"customer_inquiry": response_prompt})
    
    return response['text']

# Example usage
question = "What products do you have?"
response = answer_question(question)
print(response)