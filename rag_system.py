"""RAG System using LangChain with Qdrant"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate environment variables
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")
if not QDRANT_API_KEY:
    print("WARNING: QDRANT_API_KEY not set!")

# Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://litellm.confer.today"
)

# LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://litellm.confer.today"
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant for Confer Solutions AI.

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
    3. If the answer is not available in the context, say clearly:
    "This information is not available in our current knowledge base."
    4. Keep responses professional, clear, and business-focused.
    5. Highlight benefits, use cases, and value where applicable.
    6. Prefer concise but informative answers.
    7. Do NOT mention internal metadata, vectors, embeddings, or database structure.
    8. Do NOT reference document line numbers or blob IDs.
    9. If multiple sources say similar things, combine them into one clear answer.
    10. Tone: Professional, trustworthy, and confident (FinTech / AI consulting style).

    Your goal is to help users understand Confer Solutions AI offerings, benefits, and expertise accurately.
    Answer the following question using only the provided context.

    Context:
    {context}

    Question:
    {input}

    Answer:"""
)

# Chain
chain = prompt | llm | StrOutputParser()

# API
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "openai_key_set": bool(OPENAI_API_KEY),
        "qdrant_key_set": bool(QDRANT_API_KEY)
    }


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        # Check API keys
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        if not QDRANT_API_KEY:
            raise HTTPException(status_code=500, detail="QDRANT_API_KEY not configured")
        
        # Get embedding for the query
        vector = embeddings.embed_query(request.query)
        
        # Search Qdrant
        response = requests.post(
            "https://qdrant.confersolutions.ai/collections/confer-website/points/search",
            headers={"api-key": QDRANT_API_KEY},
            json={"vector": vector, "limit": 5, "with_payload": True},
            timeout=60
        )
        
        # Check response
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Qdrant error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Format context from retrieved documents
        context = "\n\n".join([r["payload"].get("content", "") for r in result.get("result", [])])
        
        # Generate answer using LangChain
        answer = chain.invoke({"context": context, "input": request.query})
        
        return {"answer": answer}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
