import warnings
warnings.filterwarnings("ignore")

from langgraph.prebuilt import create_react_agent as create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agents.extraction_agent import ExtractionAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent

extraction_agent = ExtractionAgent()
analysis_agent = AnalysisAgent()
evaluation_agent = EvaluationAgent()


@tool
def call_extraction_agent(request: str) -> str:
    """Send an incident report to the Extraction Agent to extract and store structured metadata."""
    return extraction_agent.run(request)


@tool
def call_analysis_agent(request: str) -> str:
    """Send an analysis request to the Analysis Agent to generate incident-level charts: MTTR/duration of incidents, service category distribution, or root cause distribution."""
    return analysis_agent.run(request)


@tool
def call_evaluation_agent(request: str) -> str:
    """Send an evaluation request to the Evaluation Agent to display extraction accuracy, extraction latency/speed, or token usage/cost results."""
    return evaluation_agent.run(request)


REPORT_WORD_THRESHOLD = 100

ORCHESTRATOR_PROMPT = (
    "You are an Incident Analysis Orchestrator. Route each user request to exactly one specialist agent:\n\n"
    "1. call_analysis_agent  — charts/analysis about the INCIDENTS themselves: duration/MTTR of incidents, "
    "service category distribution, or root cause distribution\n"
    "2. call_evaluation_agent — anything about how well the EXTRACTION PIPELINE performed: accuracy, "
    "extraction latency/speed (how long extraction took, not incident duration), or token usage/cost\n\n"
    "'latency' and 'tokens' always mean call_evaluation_agent, never call_analysis_agent — "
    "they measure the extraction process, not the incidents.\n\n"
    "Pass the user's full message to the chosen agent. Never call more than one agent per turn.\n\n"
    "After calling a specialist agent, relay its response to the user EXACTLY as returned — do not summarize, shorten, or paraphrase it."
)


def _route(user_input: str, orchestrator, history: list[dict]) -> str:
    if len(user_input.split()) > REPORT_WORD_THRESHOLD:
        return extraction_agent.run(user_input)
    history.append({"role": "user", "content": user_input})
    result = orchestrator.invoke({"messages": history})
    reply = result["messages"][-1].content
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    orchestrator = create_agent(
        model,
        [call_analysis_agent, call_evaluation_agent],
        prompt=ORCHESTRATOR_PROMPT,
    )
    history: list[dict] = []

    print("=== Incident Analysis Service (Multi-Agent) ===")
    print("Commands: paste a report to extract it, or ask for analysis/chart.")
    print("Type 'reset' to clear conversation history, 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "reset":
            history.clear()
            extraction_agent.reset()
            analysis_agent.reset()
            evaluation_agent.reset()
            print("History cleared.\n")
            continue
        if not user_input:
            continue

        print(f"\nAgent: {_route(user_input, orchestrator, history)}\n")


if __name__ == "__main__":
    main()
