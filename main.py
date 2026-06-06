from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import httpx
import json
from supabase import create_client
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "ai-hub"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class DumpInput(BaseModel):
    content: str
    source: Optional[str] = "manual"


class ChatInput(BaseModel):
    message: str
    conversation_id: Optional[str] = None


async def call_openrouter(messages: list, model: str = "openai/gpt-4o-mini") -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def get_embedding(text: str) -> list:
    """Get embedding via OpenRouter or fallback to simple hash-based vector."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "openai/text-embedding-3-small", "input": text},
            timeout=30,
        )
        data = response.json()
        return data["data"][0]["embedding"]


@app.get("/")
async def root():
    return {"status": "AI Hub is running"}


@app.post("/dump")
async def dump_info(payload: DumpInput):
    """Dump any info — AI auto-tags, categorizes, and stores it."""
    content = payload.content

    # Ask AI to categorize and summarize
    analysis = await call_openrouter([
        {"role": "system", "content": "You are an AI that analyzes and categorizes information. Respond ONLY with valid JSON, no markdown, no backticks."},
        {"role": "user", "content": f"""Analyze this and return JSON with these fields:
- title: short title (max 8 words)
- category: one of [idea, note, task, research, link, memory, other]
- tags: array of 3-5 relevant tags
- summary: one sentence summary

Content: {content}"""}
    ])

    try:
        clean = analysis.strip().replace("```json", "").replace("```", "").strip()
        meta = json.loads(clean)
    except Exception:
        meta = {"title": content[:50], "category": "note", "tags": [], "summary": content[:100]}

    # Store in Supabase
    record = {
        "content": content,
        "title": meta.get("title", ""),
        "category": meta.get("category", "note"),
        "tags": meta.get("tags", []),
        "summary": meta.get("summary", ""),
        "source": payload.source,
    }
    result = supabase.table("memories").insert(record).execute()
    record_id = result.data[0]["id"]

    # Store embedding in Pinecone
    try:
        embedding = await get_embedding(content)
        index.upsert(vectors=[{
            "id": str(record_id),
            "values": embedding,
            "metadata": {"title": meta.get("title", ""), "category": meta.get("category", ""), "summary": meta.get("summary", "")}
        }])
    except Exception as e:
        print(f"Embedding error: {e}")

    return {"id": record_id, "meta": meta}


@app.get("/memories")
async def get_memories(category: Optional[str] = None, limit: int = 50):
    """Get all stored memories."""
    query = supabase.table("memories").select("*").order("created_at", desc=True).limit(limit)
    if category:
        query = query.eq("category", category)
    result = query.execute()
    return result.data


@app.post("/chat")
async def chat(payload: ChatInput):
    """Chat with your 3rd brain — searches your memories and answers."""
    question = payload.message

    # Search Pinecone for relevant memories
    context_memories = []
    try:
        embedding = await get_embedding(question)
        results = index.query(vector=embedding, top_k=5, include_metadata=True)
        for match in results.matches:
            if match.score > 0.5:
                context_memories.append(match.metadata.get("summary", ""))
    except Exception as e:
        print(f"Search error: {e}")

    context = "\n".join(context_memories) if context_memories else "No relevant memories found."

    answer = await call_openrouter([
        {"role": "system", "content": f"""You are the user's personal AI brain. You have access to their stored memories and knowledge.

Relevant memories:
{context}

Answer based on their stored knowledge when possible. Be concise and helpful."""},
        {"role": "user", "content": question}
    ])

    return {"answer": answer, "sources_used": len(context_memories)}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    supabase.table("memories").delete().eq("id", memory_id).execute()
    try:
        index.delete(ids=[str(memory_id)])
    except Exception:
        pass
    return {"deleted": memory_id}
