import os
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from pinecone import Pinecone

app = FastAPI(title="AI Command Center", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
PINECONE_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "aihub")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")        # optional – for ClawSweeper/gitcrawl
ETSY_KEY       = os.getenv("ETSY_API_KEY", "")        # optional – for Etsy agent

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(PINECONE_INDEX)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "openai/gpt-4o"

# ── Shared AI helper ──────────────────────────────────────────────────────────
async def ask_ai(system: str, user: str, model: str = MODEL) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ]}
        )
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def embed(text: str) -> list:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": "openai/text-embedding-3-small", "input": text[:8000]}
        )
        return r.json()["data"][0]["embedding"]

# ── Models ────────────────────────────────────────────────────────────────────
class DumpRequest(BaseModel):
    content: str
    source: str = "manual"

class AgentRequest(BaseModel):
    agent: str
    input: str
    context: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class SiteEditRequest(BaseModel):
    instruction: str
    current_html: str

class EtsyRequest(BaseModel):
    task: str  # "listings" | "pricing" | "messages" | "trends"
    data: Optional[str] = None

class GitRequest(BaseModel):
    repo: str  # e.g. "owner/repo"
    task: str  # "triage" | "summary" | "issues"

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI Command Center v3 online", "agents": 15}

@app.get("/health")
def health():
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}

# ── DUMP (auto-sort into 3rd brain) ──────────────────────────────────────────
@app.post("/dump")
async def dump(req: DumpRequest, bg: BackgroundTasks):
    # 1. AI categorise + tag + summarise
    meta_raw = await ask_ai(
        "You are a 3rd-brain organiser. Return ONLY valid JSON with keys: title, category, tags (array), summary.",
        f"Analyse and categorise this content:\n\n{req.content}"
    )
    try:
        meta = json.loads(meta_raw.strip().strip("```json").strip("```"))
    except Exception:
        meta = {"title": "Note", "category": "uncategorised", "tags": [], "summary": req.content[:200]}

    # 2. Fact-check
    fact = await ask_ai(
        "You are a fact-checker. Rate the factual reliability of this content 1-10 and list any concerns. Be brief.",
        req.content[:2000]
    )

    # 3. Save to Supabase
    row = supabase.table("memories").insert({
        "content": req.content,
        "title": meta.get("title", "Untitled"),
        "category": meta.get("category", "general"),
        "tags": meta.get("tags", []),
        "summary": meta.get("summary", ""),
        "source": req.source,
    }).execute()

    # 4. Embed + upsert to Pinecone (background)
    async def _embed_and_store():
        try:
            vec = await embed(req.content)
            mem_id = str(row.data[0]["id"]) if row.data else "unknown"
            index.upsert(vectors=[{
                "id": f"mem-{mem_id}",
                "values": vec,
                "metadata": {"title": meta.get("title"), "category": meta.get("category"), "summary": meta.get("summary", "")[:300]}
            }])
        except Exception as e:
            print(f"Pinecone error: {e}")

    bg.add_task(_embed_and_store)

    return {
        "saved": True,
        "meta": meta,
        "fact_check": fact,
        "id": row.data[0]["id"] if row.data else None
    }

# ── MEMORIES ─────────────────────────────────────────────────────────────────
@app.get("/memories")
def get_memories(limit: int = 50, category: Optional[str] = None):
    q = supabase.table("memories").select("*").order("created_at", desc=True).limit(limit)
    if category:
        q = q.eq("category", category)
    return q.execute().data

@app.delete("/memories/{mem_id}")
def delete_memory(mem_id: int):
    supabase.table("memories").delete().eq("id", mem_id).execute()
    return {"deleted": True}

# ── CHAT with 3rd brain ───────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    # Semantic search
    try:
        vec = await embed(req.message)
        results = index.query(vector=vec, top_k=5, include_metadata=True)
        context_chunks = [m.metadata.get("summary", "") for m in results.matches if m.metadata]
        context = "\n".join(context_chunks)
    except Exception:
        context = ""

    system = f"""You are the user's personal AI assistant with access to their 3rd brain knowledge base.
Use the following relevant memories to answer their question:
{context}
If memories aren't relevant, answer from your own knowledge."""

    messages = [{"role": "system", "content": system}]
    for h in (req.history or []):
        messages.append(h)
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": MODEL, "messages": messages}
        )
        reply = r.json()["choices"][0]["message"]["content"]

    return {"reply": reply, "context_used": bool(context)}

# ── AGENT DISPATCH ────────────────────────────────────────────────────────────
AGENT_PROMPTS = {
    "researcher": "You are an expert research agent. Search-style: thorough, sourced, structured. Provide a deep research report with key findings, evidence, and a conclusion.",
    "project_manager": "You are a senior project manager. Break down goals into milestones, tasks, owners, and timelines. Return structured markdown.",
    "coder": "You are an elite software engineer. Write clean, production-grade code with comments and error handling.",
    "analyst": "You are a data analyst. Analyse data, find patterns, create insights, and make recommendations backed by numbers.",
    "social_monitor": "You are a social media strategist. Analyse trends, mentions, sentiment, and recommend content strategy.",
    "email_drafter": "You are a professional copywriter. Draft clear, compelling emails with subject lines. Match the requested tone.",
    "brainstormer": "You are a creative ideation expert. Generate 10-20 bold, diverse, actionable ideas. Think laterally.",
    "spending_manager": "You are a personal finance advisor. Categorise expenses, identify patterns, flag anomalies, and recommend savings.",
    "message_manager": "You are a communications expert. Draft clear, appropriate messages. Match tone to context.",
    "organiser": "You are a master organiser running 24/7. Sort, tag, link, and prioritise ALL information into the 3rd brain system. Be systematic.",
    "fact_checker": "You are a rigorous fact-checker. Verify each claim, rate reliability 1-10, cite counter-evidence. Be critical.",
    "checkin": "You are a personal AI coach. Give a warm daily check-in: review recent activity, ask smart questions, set daily intentions, flag anything needing attention.",
    "etsy_manager": "You are an Etsy business expert. Help with listings, SEO titles/tags, pricing strategy, customer response templates, and trend analysis.",
    "memory_agent": "You are a knowledge graph agent. Connect dots across information: find patterns, contradictions, links, and insights across the entire knowledge base.",
    "commander": "You are an AI mission commander. Given a goal, break it into sub-tasks, assign the best agent to each, and write an execution plan with expected outputs.",
}

@app.post("/agent/{agent_name}")
async def run_agent(agent_name: str, req: AgentRequest):
    if agent_name not in AGENT_PROMPTS:
        raise HTTPException(404, f"Agent '{agent_name}' not found. Available: {list(AGENT_PROMPTS.keys())}")

    system = AGENT_PROMPTS[agent_name]
    user_input = req.input
    if req.context:
        user_input = f"Context:\n{req.context}\n\nTask:\n{req.input}"

    result = await ask_ai(system, user_input)

    # Auto-save important agent results to 3rd brain
    if agent_name in ["researcher", "analyst", "fact_checker", "memory_agent"]:
        try:
            supabase.table("memories").insert({
                "content": result,
                "title": f"[{agent_name.upper()}] {req.input[:60]}",
                "category": "agent_output",
                "tags": [agent_name],
                "summary": result[:300],
                "source": f"agent:{agent_name}",
            }).execute()
        except Exception:
            pass

    return {
        "agent": agent_name,
        "input": req.input,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents")
def list_agents():
    return {name: {"description": prompt[:80] + "..."} for name, prompt in AGENT_PROMPTS.items()}

# ── COMMANDER – multi-agent orchestration ─────────────────────────────────────
@app.post("/commander")
async def commander_run(req: AgentRequest):
    plan_raw = await ask_ai(
        "You are the Commander. Given a mission, output ONLY JSON: {mission, steps: [{agent, task}]}. Use agents from: " + ", ".join(AGENT_PROMPTS.keys()),
        f"Mission: {req.input}"
    )
    try:
        plan = json.loads(plan_raw.strip().strip("```json").strip("```"))
    except Exception:
        return {"error": "Commander could not parse plan", "raw": plan_raw}

    results = []
    for step in plan.get("steps", [])[:5]:  # cap at 5 parallel tasks
        agent = step.get("agent", "brainstormer")
        task  = step.get("task", req.input)
        if agent in AGENT_PROMPTS:
            res = await ask_ai(AGENT_PROMPTS[agent], task)
            results.append({"agent": agent, "task": task, "result": res})

    return {"plan": plan, "results": results}

# ── SITE EDITOR agent (live HTML patch) ──────────────────────────────────────
@app.post("/site-editor")
async def site_editor(req: SiteEditRequest):
    result = await ask_ai(
        """You are a live site editor agent. Given HTML and an instruction, return ONLY the modified HTML — no explanation, no markdown fences, just the raw HTML.
Make surgical edits: only change what was asked. Preserve all existing classes, IDs, and scripts.""",
        f"INSTRUCTION: {req.instruction}\n\nCURRENT HTML:\n{req.current_html[:15000]}"
    )
    return {"html": result}

# ── OPENCLAW: ClawSweeper-style GitHub triage ─────────────────────────────────
@app.post("/github/triage")
async def github_triage(req: GitRequest):
    if not GITHUB_TOKEN:
        # If no token, just use AI to analyse whatever data user provides
        result = await ask_ai(
            "You are a GitHub issue triage agent (ClawSweeper-style). Analyse the provided issues and recommend: CLOSE, KEEP, or ESCALATE for each. Give reasons.",
            f"Repo: {req.repo}\nTask: {req.task}\nData: {req.data or 'No data provided – give general triage guidance'}"
        )
        return {"result": result, "note": "Add GITHUB_TOKEN env var to enable live GitHub API access"}

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"https://api.github.com/repos/{req.repo}/issues?state=open&per_page=20", headers=headers)
        issues = r.json()

    issues_text = "\n".join([f"#{i['number']}: {i['title']} ({i.get('comments',0)} comments)" for i in issues[:20]])
    result = await ask_ai(
        "You are ClawSweeper, a GitHub triage agent. For each issue: recommend CLOSE (stale/duplicate/out-of-scope), KEEP (valid), or ESCALATE (critical). Give brief reasons.",
        f"Repo: {req.repo}\n\nOpen Issues:\n{issues_text}"
    )
    return {"repo": req.repo, "triage": result, "issue_count": len(issues)}

# ── OPENCLAW: Etsy Manager ────────────────────────────────────────────────────
@app.post("/etsy")
async def etsy_manager(req: EtsyRequest):
    task_prompts = {
        "listings": "Optimise this Etsy listing for SEO: generate a compelling title (max 140 chars), 13 tags, description with keywords, and pricing suggestions.",
        "pricing": "Analyse this Etsy pricing situation and recommend optimal pricing strategy considering competition, perceived value, and profit margin.",
        "messages": "Draft a professional, warm, on-brand Etsy customer message response. Resolve their concern while maintaining seller reputation.",
        "trends": "Analyse current Etsy market trends for this niche. What's selling, what tags are trending, what should I create next?"
    }
    system = task_prompts.get(req.task, task_prompts["listings"])
    result = await ask_ai(system, req.data or f"Task: {req.task}")
    return {"task": req.task, "result": result}

# ── CHECKIN agent (daily summary) ─────────────────────────────────────────────
@app.get("/checkin")
async def daily_checkin():
    # Get recent memories
    recent = supabase.table("memories").select("title,category,created_at").order("created_at", desc=True).limit(10).execute().data
    summary_input = f"Recent brain entries:\n" + "\n".join([f"- [{r['category']}] {r['title']}" for r in recent])

    result = await ask_ai(AGENT_PROMPTS["checkin"], summary_input)
    return {"checkin": result, "timestamp": datetime.utcnow().isoformat()}

# ── SEARCH memories ───────────────────────────────────────────────────────────
@app.post("/search")
async def search_memories(req: ChatRequest):
    try:
        vec = await embed(req.message)
        results = index.query(vector=vec, top_k=8, include_metadata=True)
        hits = [{"score": round(m.score, 3), **m.metadata} for m in results.matches]
        return {"query": req.message, "results": hits}
    except Exception as e:
        return {"error": str(e)}

# ── STATS ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    total    = supabase.table("memories").select("id", count="exact").execute()
    by_cat   = supabase.rpc("get_category_counts") if False else []  # optional RPC
    return {
        "total_memories": total.count,
        "agents_available": len(AGENT_PROMPTS),
        "openclaw_tools": ["ClawSweeper/github-triage", "Etsy-Manager", "site-editor", "fact_checker", "organiser", "commander"],
        "timestamp": datetime.utcnow().isoformat()
    }
