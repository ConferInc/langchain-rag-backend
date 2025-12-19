
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load env vars
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_base="https://litellm.confer.today"
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),    # Reads from .env
    port=443,
    api_key=os.getenv("QDRANT_API_KEY"), # Reads from .env
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="confer-website",
    embedding=embeddings,
    content_payload_key="content",
)

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    openai_api_base="https://litellm.confer.today",
    temperature=0,
)

# Setup RAG Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for Confer Solutions AI.
Answer questions ONLY using the provided context.
Rules:
1. Use ONLY the retrieved context.
2. Do NOT hallucinate.
3. If answer is missing, say: "This information is not available in our current knowledge base."
4. Be professional and concise.

Context:
{context}"""),
    ("human", "{input}"),
])

retrieval_chain = create_retrieval_chain(
    vector_store.as_retriever(search_kwargs={"k": 5}),
    create_stuff_documents_chain(llm, prompt)
)

# FastAPI App
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    return {"answer": retrieval_chain.invoke({"input": request.question})["answer"]}

if __name__ == "__main__":
    print("Starting Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
