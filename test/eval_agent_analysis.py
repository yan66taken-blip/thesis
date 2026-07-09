"""
Analysis Agent test harness.

Metrics:
  - Tool selection accuracy    (correct plotting tool among 9 tools)
  - Parameter accuracy         (vendor/year filters)

All tools run in stub mode: no file I/O or chart rendering.

Usage:
    python test/eval_agent_analysis.py
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.prebuilt import create_react_agent as create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.analysis_agent import SYSTEM_PROMPT as ANALYSIS_AGENT_PROMPT


CASES_PATH = Path(__file__).parent / "analysis_agent_test_cases.json"
RESULTS_PATH = Path(__file__).parent / "analysis_agent_results.json"


# ── Pydantic schema for stub tools ─────────────────────────────────────

class PlotArgs(BaseModel):
    output_path: Optional[str] = None
    vendors: Optional[list[str]] = None
    years: Optional[list[int]] = None


# ── Stub tools ──────────────────────────────────────────────────────────

def _stub(name: str, description: str) -> StructuredTool:
    return StructuredTool(
        name=name,
        description=description,
        func=lambda **_: "stub",
        args_schema=PlotArgs,
    )

ANALYSIS_TOOLS = [
    _stub("plot_duration_box",
          "Plots a log-scale MTTR boxplot showing incident duration distribution. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_duration_bar",
          "Plots a bar chart of mean MTTR per year. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_duration_line",
          "Plots a line chart of monthly mean MTTR trend over time. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_rootcause_pie",
          "Plots a pie chart of root cause category distribution. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_rootcause_bar",
          "Plots a horizontal bar chart of root cause category percentages. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_rootcause_stacked",
          "Plots a 100% stacked bar chart comparing root cause distribution across vendors. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_servicecategory_pie",
          "Plots a pie chart of service category distribution. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_servicecategory_bar",
          "Plots a horizontal bar chart of service category percentage distribution. "
          "Optionally filter by vendors and/or years."),
    _stub("plot_servicecategory_stacked",
          "Plots a 100% stacked bar chart comparing service category distribution across vendors. "
          "Optionally filter by vendors and/or years."),
]


# ── Agent builder ───────────────────────────────────────────────────────

def build_analysis_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(model, ANALYSIS_TOOLS, prompt=ANALYSIS_AGENT_PROMPT)


# ── Tool call extraction ───────────────────────────────────────────────

def get_tool_calls(messages: list) -> list[dict]:
    calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                calls.append({"name": tc["name"], "args": tc.get("args", {})})
    return calls


def invoke(agent, query: str) -> list[dict]:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return get_tool_calls(result["messages"])


# ── Evaluation functions ────────────────────────────────────────────────

def eval_tool_selection(cases: list, agent) -> dict:
    correct, rows = 0, []
    for c in cases:
        calls = invoke(agent, c["input"])
        predicted_names = [tc["name"] for tc in calls] if calls else []
        expected = c["expected_tool"]

        if isinstance(expected, list):
            match = set(expected) == set(predicted_names)
        else:
            match = predicted_names[:1] == [expected]

        correct += int(match)
        rows.append({
            "input": c["input"][:80],
            "expected": expected,
            "predicted": predicted_names if isinstance(expected, list) else (predicted_names[0] if predicted_names else "none"),
            "match": match,
        })
    return {"accuracy": correct / len(cases), "n": len(cases), "details": rows}


def eval_params(cases: list, agent) -> dict:
    correct, rows = 0, []
    for c in cases:
        calls = invoke(agent, c["input"])
        args = calls[0]["args"] if calls else {}
        expected = c.get("expected_params", {})

        if not expected:
            match = True
        else:
            match = all(
                _param_match(args.get(k), v)
                for k, v in expected.items()
            )

        correct += int(match)
        rows.append({
            "input": c["input"][:80],
            "expected_tool": c.get("expected_tool", ""),
            "expected_params": expected,
            "predicted_tool": calls[0]["name"] if calls else "none",
            "predicted_params": args,
            "match": match,
        })
    return {"accuracy": correct / len(cases), "n": len(cases), "details": rows}


def _param_match(actual, expected) -> bool:
    if expected is None:
        return actual is None or actual == []
    if isinstance(expected, list):
        return set(actual or []) == set(expected)
    return actual == expected


# ── Pretty printing ────────────────────────────────────────────────────

def _print_section(title: str, result: dict):
    n = result["n"]
    correct = round(result["accuracy"] * n)
    pct = result["accuracy"] * 100
    print(f"\n{title}  —  {correct}/{n}  ({pct:.1f}%)")
    print("─" * 72)
    for r in result["details"]:
        tag = "OK  " if r["match"] else "FAIL"
        print(f"  [{tag}]  {r['input']}")
        if not r["match"]:
            exp = r.get("expected") or r.get("expected_params")
            got = r.get("predicted") or r.get("predicted_params")
            print(f"           expected → {exp}")
            print(f"           got      → {got}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    cases = json.loads(CASES_PATH.read_text())

    print("Building stub analysis agent...")
    agent = build_analysis_agent()

    print("Running analysis agent test cases...\n")

    results = {
        "tool_selection": eval_tool_selection(
            cases["analysis_tool_selection"], agent
        ),
        "params": eval_params(
            cases["analysis_params"], agent
        ),
        "boundary": eval_tool_selection(
            cases["analysis_boundary"], agent
        ),
    }

    labels = {
        "tool_selection": "Analysis agent tool selection",
        "params":         "Analysis agent parameter accuracy",
        "boundary":       "Analysis agent boundary cases",
    }

    print("=" * 72)
    print("ANALYSIS AGENT TEST SUMMARY")
    print("=" * 72)
    print(f"  {'Metric':<45} {'Score':<10} Accuracy")
    print("  " + "─" * 65)
    for key, label in labels.items():
        r = results[key]
        n = r["n"]
        correct = round(r["accuracy"] * n)
        print(f"  {label:<45} {correct}/{n:<8}   {r['accuracy']*100:.1f}%")

    for key, label in labels.items():
        _print_section(label, results[key])

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDetailed results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
