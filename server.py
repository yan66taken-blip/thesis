import base64, os, time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib
matplotlib.use("Agg")  # headless backend, REQUIRED

from langgraph.prebuilt import create_react_agent as create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agents.extraction_agent import ExtractionAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent

# --- Orchestrator setup (mirrors main.py) ---
extraction_agent = ExtractionAgent()
analysis_agent = AnalysisAgent()
evaluation_agent = EvaluationAgent()


@tool
def call_extraction_agent(request: str) -> str:
    """Send an incident report to the Extraction Agent to extract and store structured metadata."""
    return extraction_agent.run(request)


@tool
def call_analysis_agent(request: str) -> str:
    """Send an analysis request to the Analysis Agent to generate MTTR, service category, or root cause charts."""
    return analysis_agent.run(request)


@tool
def call_evaluation_agent(request: str) -> str:
    """Send an evaluation request to the Evaluation Agent to display extraction accuracy results."""
    return evaluation_agent.run(request)


ORCHESTRATOR_PROMPT = (
    "You are an Incident Analysis Orchestrator. Route each user request to exactly one specialist agent:\n\n"
    "1. call_extraction_agent — if the input is an incident report (long text describing an outage or service disruption)\n"
    "2. call_analysis_agent  — if the request asks for charts or analysis: duration/MTTR, service category, or root cause\n"
    "3. call_evaluation_agent — if the request mentions evaluation, accuracy, or eval results\n\n"
    "Pass the user's full message to the chosen agent. Never call more than one agent per turn.\n\n"
    "After calling a specialist agent, relay its response to the user EXACTLY as returned — do not summarize, shorten, or paraphrase it."
)

_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
orchestrator = create_agent(
    model=_model,
    tools=[call_extraction_agent, call_analysis_agent, call_evaluation_agent],
    prompt=ORCHESTRATOR_PROMPT,
)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev only
    allow_methods=["*"],
    allow_headers=["*"],
)


_history: list[dict] = []


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
def chat(body: ChatIn):
    global _history
    _history.append({"role": "user", "content": body.message})
    t_before = time.time()
    result = orchestrator.invoke({"messages": _history})
    reply_text = result["messages"][-1].content
    _history.append({"role": "assistant", "content": reply_text})

    # Find any PNG written/updated during this request
    image_b64 = None
    image_description = None
    png_candidates = [f for f in os.listdir(".") if f.endswith(".png") and os.path.getmtime(f) >= t_before]
    if png_candidates:
        latest = max(png_candidates, key=os.path.getmtime)
        with open(latest, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        vision_msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this chart in 2-3 sentences, focusing on the key insights."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ])
        image_description = _model.invoke([vision_msg]).content

    return {"reply": reply_text, "image": image_b64, "image_description": image_description}


@app.post("/reset")
def reset():
    global _history
    _history.clear()
    extraction_agent.reset()
    analysis_agent.reset()
    evaluation_agent.reset()
    return {"status": "ok"}
