from langgraph.prebuilt import create_react_agent as create_agent
from langchain_openai import ChatOpenAI
from tools.plot_eval import (
    report_eval_accuracy,
    plot_eval_latency,
    plot_eval_tokens,
)

SYSTEM_PROMPT = (
    "You are an Evaluation Agent that reports extraction quality metrics.\n\n"
    "Tools:\n"
    "- report_eval_accuracy — markdown accuracy table across all vendors for categorical labels "
    "(service_category, user_symptom_category, root_cause_category), datetime labels "
    "(start_time, end_time matched at HH:MM), and service_name (exact match). "
    "Use for accuracy numbers, summary, or comparisons.\n"
    "- plot_eval_latency    — latency box plot, one box per vendor\n"
    "- plot_eval_tokens     — avg token usage (total vs cached), one group per vendor\n\n"
    "All tools accept optional 'vendors' (e.g. ['Azure','AWS']) and 'years' (e.g. [2022,2023]) filters.\n\n"
    "Default paths (the final evaluation, see eval_final/README.md):\n"
    "  Azure → eval_final/Azure.csv\n"
    "  AWS   → eval_final/AWS.csv\n"
    "  GCP   → eval_final/GCP.csv\n"
    "  All vendors → eval_dir='eval_final'\n\n"
    "Routing rules:\n"
    "- 'accuracy', 'results', 'numbers', 'table', 'all providers', 'summary' → report_eval_accuracy\n"
    "- 'latency', 'speed', 'time per report' → plot_eval_latency\n"
    "- 'tokens', 'cost', 'cache' → plot_eval_tokens\n\n"
    "After calling report_eval_accuracy, relay the full markdown table verbatim, "
    "then add a one-sentence interpretation focusing on categorical and datetime accuracy."
)

ALL_TOOLS = [
    report_eval_accuracy,
    plot_eval_latency,
    plot_eval_tokens,
]


class EvaluationAgent:
    def __init__(self):
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._agent = create_agent(model, ALL_TOOLS, prompt=SYSTEM_PROMPT)
        self._history: list[dict] = []

    def run(self, query: str) -> str:
        self._history.append({"role": "user", "content": query})
        result = self._agent.invoke({"messages": self._history})
        reply = result["messages"][-1].content
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self._history.clear()
