from langgraph.prebuilt import create_react_agent as create_agent
from langchain_openai import ChatOpenAI
from tools.extractor import extractor, smart_extractor

SYSTEM_PROMPT = (
    "You are an Extraction Agent specialized in parsing cloud incident reports.\n\n"
    "You have two tools:\n"
    "1. smart_extractor — use when the user asks for SPECIFIC fields from a report "
    "(e.g. 'what is the root cause?', 'extract only the service category and start time'). "
    "Pass the report and a list of the requested field names. It returns only those fields instantly "
    "while a full extraction runs in the background to persist the complete record.\n"
    "2. extractor — use when the user provides a report with no specific field request, "
    "or wants a full summary. It extracts and stores all fields.\n\n"
    "Valid field names: vendor, service_name, service_category, start_time, end_time, "
    "user_symptom, user_symptom_category, root_cause, root_cause_category.\n\n"
    "After calling either tool, respond naturally and informatively — never dump raw JSON."
)


class ExtractionAgent:
    def __init__(self):
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._agent = create_agent(model, [extractor, smart_extractor], prompt=SYSTEM_PROMPT)
        self._history: list[dict] = []

    def run(self, query: str) -> str:
        self._history.append({"role": "user", "content": query})
        result = self._agent.invoke({"messages": self._history})
        reply = result["messages"][-1].content
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self._history.clear()
