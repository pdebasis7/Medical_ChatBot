from flask import Flask, render_template, jsonify, request
from src.helper import txt_embeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.chat_models import AzureChatOpenAI
import openai
from src.helper import load_pdf, text_splitter, txt_embeddings

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')



extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embeddings = txt_embeddings()



persist_directory = 'db'

#Creating Embeddings for Each of The Text Chunks & storing
#vectordb = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)


#vectordb.persist()

vectordb = None

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=AzureChatOpenAI( 
    #deployment_name=DEPLOYMENT_NAME,
    #model_kwargs={'deployment_name':'omsgenai1'},
    deployment_name='omsgenai1',
    openai_api_key='1ea71466d8ac49f4999b1984ffbab0a1',
    openai_api_version="2023-07-01-preview",
    openai_api_base='https://omsgenai.openai.azure.com/',
    temperature=0.5

)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="localhost", port= 5050, debug= True)