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
#test
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
    ("system", """
You are the AI Assistant for Confer Solutions AI.
Your goal is to be a helpful, knowledgeable, and engaging partner for users interested in Confer Solutions' products, lending capabilities, and workflow automation.

The knowledge base available to you includes:
- Company mission and overview
- AI-powered lending and mortgage solutions
- Income validation and document processing details
- Press releases, blogs, and case studies

CORE INSTRUCTIONS:
1. **Source of Truth:** Base your business and technical answers *primarily* on the provided context.
2. **Tone:** Professional, warm, and consultative. Avoid robotic or purely transactional language.
3. **Accuracy:** Do not make up specific facts about Confer Solutions features or pricing if they are not in the text.
4. **Follow up:** Ask follow up questions related to the question the user asks to make the conversation natural and friendly.

HANDLING GREETINGS & GENERAL CHIT-CHAT:
- If the user says "Hi", "Hello", or asks "How are you?", **you do NOT need retrieved context.**
HANDLING WEAK OR MISSING CONTEXT:
- If the retrieved context does not contain the answer, **do not say "I don't know" or "Information not available."**
- Instead, pivot helpfulness. Say something like:
  "I don't have the specific details on that right here, but I can tell you about how Confer Solutions handles [related topic from context], or I can help you find..."
- If the question is completely out of scope (e.g., "What is the weather?"), politely guide them back: "I specialize in Confer Solutions' AI and lending technology. How can I help you with that?"

META QUESTIONS (e.g., "What can you do?"):
- Summarize your capabilities based on the general topics in the knowledge base (e.g., "I can help explain our AI-powered lending platform, income validation tools, and how we automate workflows.").

Context:
{context}
"""),
    ("human", "{input}")
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
