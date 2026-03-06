
"""
RAG Contact Center Q&A - FastAPI Application
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os, time, json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- Config ---
DEMO_MODE = True
CHAT_MODEL      = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
TOP_K           = 5

app = FastAPI(title="RAG Contact Center API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading FAISS index...")
faiss_index = faiss.read_index("faiss_index.bin")
with open("chunks_metadata.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
print(f"Ready: {faiss_index.ntotal} vectors loaded")

class QueryRequest(BaseModel):
    question: str
    category: Optional[str] = None
    top_k: Optional[int] = None

class SourceDoc(BaseModel):
    chunk_id: str
    doc_name: str
    category: str
    similarity: float
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    metadata: dict

def get_embedding(text: str):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(EMBEDDING_DIM).tolist()

def search(query: str, top_k: int, cat_filter: Optional[str] = None):
    qe = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(qe)
    dists, idxs = faiss_index.search(qe, top_k * 3)
    out = []
    for d, i in zip(dists[0], idxs[0]):
        if i == -1:
            continue
        c = all_chunks[i].copy()
        if cat_filter and c.get("category") != cat_filter:
            continue
        c["similarity_score"] = float(d)
        out.append(c)
        if len(out) >= top_k:
            break
    return out

def generate(question: str, chunks: list):
    top_chunk = chunks[0]
    return f"Based on the {top_chunk['doc_name']}, {top_chunk['chunk_text'][:220].strip()}..."

def log(query: str, chunk_id: str, answer: str, ms: int):
    return

@app.get("/")
def root():
    return {"status": "online", "vectors": faiss_index.ntotal, "model": CHAT_MODEL}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "Question empty")

    t0 = time.time()
    k = int(req.top_k or TOP_K)
    chunks = search(q, k, req.category)
    if not chunks:
        raise HTTPException(404, "No relevant documents found")

    answer = generate(q, chunks)
    ms = int((time.time() - t0) * 1000)
    log(q, chunks[0]["chunk_id"], answer, ms)

    return QueryResponse(
        answer=answer,
        sources=[
            SourceDoc(
                chunk_id=c["chunk_id"],
                doc_name=c["doc_name"],
                category=c["category"],
                similarity=round(c["similarity_score"], 4),
                excerpt=c["chunk_text"][:200] + "...",
            )
            for c in chunks
        ],
        metadata={"model": CHAT_MODEL, "chunks": len(chunks), "response_ms": ms},
    )

@app.get("/documents")
def documents():
    docs = []
    seen = set()
    for c in all_chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            docs.append({
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
                "category": c["category"]
            })
    return {"total": len(docs), "documents": docs}

@app.get("/analytics")
def analytics():
    return {"stats": {"total": 0, "avg_ms": 0}, "recent": []}
