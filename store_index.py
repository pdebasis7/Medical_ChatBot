from src.helper import load_pdf, text_splitter, txt_embeddings
from langchain.vectorstores import Chroma
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()




extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embeddings = txt_embeddings()



persist_directory = 'db'

#Creating Embeddings for Each of The Text Chunks & storing
#vectordb = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)

# persiste the db to disk





vectordb = None

vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)

print(len(vectordb.get()['documents']))