from src.helper import load_pdf, text_split, download_hugging_face_embeddings
#from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone 
#pip install pinecone-client

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("/Users/rashmi/Desktop/End-to-end-Medical-Chatbot-using-Llama2/data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
#pinecone.init(api_key=PINECONE_API_KEY,
#              environment=PINECONE_API_ENV)

# Set the environment variables for Pinecone API key and environment
os.environ['PINECONE_API_KEY'] = "2a646a8b-58b8-4e28-84c9-ca66ff586687"
os.environ['PINECONE_ENV'] = "Serverless"

pinecone_client=pinecone.Pinecone(api_key=PINECONE_API_KEY,environment=PINECONE_API_ENV)


index_name="mchatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
