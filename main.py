
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
Your job is to answer user questions ONLY using the information provided in the retrieved context from the knowledge base.

The knowledge base contains:
- Confer Solutions AI company overview and mission
- AI-powered lending and mortgage solutions
- Income validation, document processing, workflow automation
- Press releases, blogs, and industry articles
- Product benefits, features, and use cases

RULES:
1. Use ONLY the retrieved context to generate answers.
2. Do NOT assume or hallucinate information outside the context.
3. If the answer is not available in the context, say clearly: "This information is not available in our current knowledge base."
4. Keep responses professional, clear, and business-focused.
5. Highlight benefits, use cases, and value where applicable.
6. Prefer concise but informative answers.
7. Do NOT mention internal metadata, vectors, embeddings, or database structure.
8. Do NOT reference document line numbers or blob IDs.
9. If multiple sources say similar things, combine them into one clear answer.
10. Tone: Professional, trustworthy, and confident (FinTech / AI consulting style).

Your goal is to help users understand Confer Solutions AI offerings, benefits, and expertise accurately.

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
