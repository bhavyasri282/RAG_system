
import os
from fastapi import FastAPI
import uvicorn

from langserve import add_routes

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# Set Gemini API Key
# -------------------------------

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# -------------------------------
# LLM
# -------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# -------------------------------
# Sample Document
# -------------------------------

text = """
Artificial Intelligence (AI) is the simulation of human intelligence by machines.

Machine Learning is a subset of Artificial Intelligence.

Deep Learning is a subset of Machine Learning.

RAG stands for Retrieval-Augmented Generation.

FAISS is a vector database used for similarity search.
"""

documents = [Document(page_content=text)]

# -------------------------------
# Split Documents
# -------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

# -------------------------------
# Embeddings
# -------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# -------------------------------
# Vector Store
# -------------------------------

vector_db = FAISS.from_documents(
    chunks,
    embeddings,
)

retriever = vector_db.as_retriever()

# -------------------------------
# Prompt
# -------------------------------

prompt = ChatPromptTemplate.from_template(
"""
Answer the question only from the context.

Context:
{context}

Question:
{question}

Answer:
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------
# FastAPI App
# -------------------------------

app = FastAPI(title="Gemini RAG API")

add_routes(
    app,
    rag_chain,
    path="/rag",
    playground_type="default",
)

@app.get("/")
def home():
    return {"message": "Gemini RAG API Running"}

# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
