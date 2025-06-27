from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from pprint import pprint

# Chroma db Client
client = chromadb.Client()

# Create a new collection or get if it exists
collection = client.get_or_create_collection(name="Nike_Ecommerce")

# Get the document
loader = PyPDFLoader(file_path="./src/documents/BI_Structure_Nike_Ecommerce_Policy.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(pages)

docs = [doc.page_content for doc in chunks]
ids = [f"chunk_{i}" for i in range(len(docs))]

# Use upsert to avoid duplicate insertions
collection.upsert(documents=docs, ids=ids)

query = "What is the warranty duration for Nike products?"
results = collection.query(
    query_texts=[query],
    n_results=1
)
# pprint((results["documents"][0]))