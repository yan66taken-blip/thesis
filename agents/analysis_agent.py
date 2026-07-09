from langgraph.prebuilt import create_react_agent as create_agent
from langchain_openai import ChatOpenAI
from tools.plot_duration import plot_duration_box, plot_duration_bar, plot_duration_line
from tools.plot_rootcause import plot_rootcause_pie, plot_rootcause_bar, plot_rootcause_stacked
from tools.plot_servicecategory import plot_servicecategory_pie, plot_servicecategory_bar, plot_servicecategory_stacked

SYSTEM_PROMPT = (
    "You are an Analysis Agent that generates incident analysis charts.\n"
    "All tools accept optional 'vendors' (e.g. ['Azure','AWS']) and 'years' (e.g. [2022,2023]) filters.\n\n"
    "TOPIC ROUTING:\n"
    "- 'root cause' / 'cause' → plot_rootcause_*\n"
    "- 'service category' / 'service distribution' → plot_servicecategory_*\n"
    "- 'duration' / 'MTTR' / 'time' → plot_duration_*\n\n"
    "CHART TYPE ROUTING (within each topic):\n"
    "- 'pie' → *_pie\n"
    "- 'bar' / 'horizontal bar' / default → *_bar\n"
    "- 'stacked' / 'by vendor' / 'compare vendors' → *_stacked\n"
    "- 'box' / 'boxplot' / 'distribution' → plot_duration_box\n"
    "- 'line' / 'trend' / 'over time' → plot_duration_line\n\n"
    "Call exactly one tool per request. After calling, relay the result to the user."
)

ALL_TOOLS = [
    plot_duration_box, plot_duration_bar, plot_duration_line,
    plot_rootcause_pie, plot_rootcause_bar, plot_rootcause_stacked,
    plot_servicecategory_pie, plot_servicecategory_bar, plot_servicecategory_stacked,
]


class AnalysisAgent:
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
