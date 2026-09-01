"""FastAPI app for the Harry Potter RAG chatbot.

Endpoints:
    GET  /         - basic service info
    GET  /health   - liveness check for the frontend's "Ping API" button
    POST /query    - routes the query (retrieve / chitchat / off-topic),
                     retrieves context from a local FAISS index when needed,
                     and generates the final answer.

The embedding model and chunk payload field names here (book_name,
page_number, content) must match what rag_pipeline.ipynb used when
building the FAISS index.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ============================= Setup =============================

load_dotenv()

app = FastAPI(title="Harry Potter RAG API")


# ============================= CORS =============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_text_content(response) -> str:
    """Convert LangChain message content into a plain string."""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        return "".join(text_parts)

    return str(content)


# ============================= Configuration =============================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DEFAULT_TOP_K = int(os.getenv("TOP_K", 3))


# ============================= Debug =============================

print(f"DEBUG: EMBEDDING_MODEL={EMBEDDING_MODEL}")
print(f"DEBUG: GEMINI_MODEL={GEMINI_MODEL}")
print(f"DEBUG: GROQ_MODEL={GROQ_MODEL}")

print(
    f"DEBUG: GEMINI_API_KEY loaded="
    f"{bool(GEMINI_API_KEY)}"
)

print(
    f"DEBUG: GROQ_API_KEY loaded="
    f"{bool(GROQ_API_KEY)}"
)


# ============================= Paths =============================

BASE_DIR = Path(__file__).resolve().parent

FAISS_INDEX_PATH = BASE_DIR / "harrypotter.index"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"


# ============================= Load Embedding Model =============================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# ============================= Load FAISS =============================

if not FAISS_INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {FAISS_INDEX_PATH}\n"
        "Run the notebook cell that creates harrypotter.index first."
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_PATH}\n"
        "Run the notebook cell that creates chunks.pkl first."
    )


print("Loading FAISS index...")

index = faiss.read_index(str(FAISS_INDEX_PATH))

print(
    f"DEBUG: FAISS index loaded successfully. "
    f"Vectors={index.ntotal}, "
    f"Dimension={index.d}"
)


print("Loading chunks...")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"DEBUG: Loaded {len(chunks)} chunks.")


# ============================= Validate FAISS =============================

if index.ntotal != len(chunks):
    raise ValueError(
        f"FAISS/chunks mismatch: "
        f"FAISS contains {index.ntotal} vectors, "
        f"but chunks.pkl contains {len(chunks)} chunks."
    )

print("DEBUG: FAISS index and chunks are aligned.")


# ============================= Load LLMs =============================

gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================


class QueryRequest(BaseModel):
    query: str

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Chunks to retrieve; defaults to TOP_K in .env",
    )


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================


@app.get("/")
def root():
    return {
        "name": "Harry Potter RAG API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store": "FAISS",
        "vectors": index.ntotal,
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    # ============================================================
    # 1. QUERY ROUTING
    # ============================================================

    ROUTER_SYSTEM_PROMPT = """
    You are a query router for a Harry Potter question-answering system.

    Classify the user's query into exactly ONE of these categories:

    - retrieve: The user is asking a question about the Harry Potter books
      or information that should be answered using the stored book content.

    - chitchat: The user is making casual conversation, greeting, thanking,
      or asking something unrelated to retrieving book information but still
      conversational.

    - off-topic: The user is asking for information unrelated to the Harry
      Potter books and not simply engaging in casual conversation.

    Return ONLY ONE of these words:

    retrieve
    chitchat
    off-topic

    Do not provide explanations, punctuation, or any other text.
    """

    route = (
        groq_llm.invoke(
            [
                SystemMessage(
                    content=ROUTER_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=request.query
                ),
            ]
        )
        .content
        .strip()
        .lower()
    )


    if route not in {
        "retrieve",
        "chitchat",
        "off-topic",
    }:
        route = "off-topic"


    # ============================================================
    # 2. CHITCHAT
    # ============================================================

    if route == "chitchat":

        CHITCHAT_SYSTEM_PROMPT = """
        You are a friendly Harry Potter assistant.

        Respond naturally and briefly to casual conversation,
        greetings, thanks, and simple friendly questions.

        Keep the tone warm and conversational.

        Do not answer questions that require information
        from the Harry Potter books.
        """

        response = groq_llm.invoke(
            [
                SystemMessage(
                    content=CHITCHAT_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=request.query
                ),
            ]
        )

        return QueryResponse(
            query=request.query,
            route=route,
            answer=get_text_content(response),
            sources=[],
        )


    # ============================================================
    # 3. OFF-TOPIC
    # ============================================================

    if route == "off-topic":

        return QueryResponse(
            query=request.query,
            route=route,
            answer=(
                "I can only answer questions about "
                "the Harry Potter books."
            ),
            sources=[],
        )


    # ============================================================
    # 4. EMBEDDING
    # ============================================================

    # E5 models expect:
    #
    # passage: ...  during indexing
    # query: ...    during retrieval

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0]

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    ).reshape(1, -1)


    # ============================================================
    # 5. FAISS RETRIEVAL
    # ============================================================

    top_k = request.top_k or DEFAULT_TOP_K

    # Never request more vectors than actually exist
    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_vector,
        top_k,
    )


    # ============================================================
    # 6. BUILD RETRIEVED RESULTS
    # ============================================================

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        # FAISS can return -1 when there aren't enough results
        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        results.append(
            {
                "score": float(score),
                "payload": chunk,
            }
        )


    # ============================================================
    # 7. BUILD RAG CONTEXT
    # ============================================================

    context = "\n\n".join(
        f"Book: {result['payload']['book_name']}\n"
        f"Page: {result['payload']['page_number']}\n"
        f"Content: {result['payload']['content']}"
        for result in results
    )


    # ============================================================
    # 8. RAG GENERATION
    # ============================================================

    RAG_SYSTEM_PROMPT = """
    You are a Harry Potter question-answering assistant.

    Answer the user's question using only the provided context.

    Do not use outside knowledge or make up information.

    If the context does not contain enough information to answer
    the question, say exactly: "I do not know."

    Keep the answer concise and clear.
    """


    response = gemini_llm.invoke(
        [
            SystemMessage(
                content=RAG_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question:\n{request.query}"
                )
            ),
        ]
    )


    # ============================================================
    # 9. RESPONSE
    # ============================================================

    return QueryResponse(
        query=request.query,
        route=route,
        answer=get_text_content(response),
        sources=[
            Source(
                book_name=result["payload"]["book_name"],
                page_number=result["payload"]["page_number"],
                score=result["score"],
            )
            for result in results
        ],
    )