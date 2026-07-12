# ODAgent — Multi-Agent Cloud Incident Analysis

A thesis project: a multi-agent LLM system (LangGraph + OpenAI `gpt-4o-mini`) that extracts
structured metadata from cloud incident reports, analyses incident trends, and evaluates its own
extraction pipeline.

Dataset: 460 labeled incident reports — AWS (n=150), Azure (n=95), GCP (n=215). See
[eval_final/README.md](eval_final/README.md) for the extraction accuracy results that back the
model/prompt choices used in the thesis.

## Architecture

A top-level orchestrator routes each user message to exactly one specialist agent:

| Agent | File | Tools |
|---|---|---|
| **ExtractionAgent** | [agents/extraction_agent.py](agents/extraction_agent.py) | `extractor`, `smart_extractor` — parse an incident report and persist structured fields |
| **AnalysisAgent** | [agents/analysis_agent.py](agents/analysis_agent.py) | `plot_rootcause`,`plot_servicecategory`,`plot_duration` |
| **EvaluationAgent** | [agents/evaluation_agent.py](agents/evaluation_agent.py) | `report_eval_accuracy`, `plot_eval_latency`, `plot_eval_tokens` — how well the extraction pipeline itself performs |

Two entry points expose the orchestrator:

- [main.py](main.py) — interactive CLI REPL
- [server.py](server.py) — FastAPI backend (`POST /chat`, `POST /reset`) for an external frontend

## Quick start

```bash
bash setup_env.sh            # create .venv and install dependencies (once)
source .venv/bin/activate
export OPENAI_API_KEY=sk-... # or set it in ~/.zshrc and open a new shell
python main.py                # CLI
# or
uvicorn server:app --port 8000  # HTTP API
```

Full local setup walkthrough, including common gotchas (stale shells not picking up a new API
key, restarting `uvicorn` after code changes, expected data file locations) is in
[docs/local_setup.md](docs/local_setup.md).

## Evaluation

Two kinds of evaluation live in this repo:

- **Extraction accuracy** — [evaluate.py](evaluate.py) runs the extractor against labeled ground
  truth in [label_data/](label_data/) and writes results to `eval_final/` (already computed; see
  [eval_final/README.md](eval_final/README.md) for the numbers).
- **Agent-level test harnesses** — in [test/](test/), each specialist and the orchestrator itself
  are evaluated on tool-selection / routing accuracy against a fixed set of prompts, using stub
  tools (no real file I/O or OpenAI calls beyond the router itself):
  - `python test/eval_agent_analysis.py` — AnalysisAgent tool selection + parameter accuracy
  - `python test/eval_agent_eval.py` — EvaluationAgent tool selection + parameter accuracy
  - `python test/eval_orchestrator.py` — orchestrator agent-routing accuracy (extraction vs.
    analysis vs. evaluation), the top-level metric for the multi-agent system as a whole

Each script prints a summary table and writes per-case detail to a `*_results.json` file next to
it.

## Project structure

```
agents/       ExtractionAgent, AnalysisAgent, EvaluationAgent
tools/        LangChain @tool functions used by the agents (extractor, plot_*, data_loader)
prompt/       Extraction prompt(s) used by tools/extractor.py
label_data/   Ground-truth labeled incident reports (per vendor)
eval_final/   Final extraction evaluation: predictions, match flags, charts, writeup
logs/         Live extracted records (written at runtime) + plot_log.csv (plot call log)
test/         Agent/orchestrator evaluation harnesses + their test-case and result JSON files
docs/         Local setup walkthrough
main.py       CLI entry point
server.py     FastAPI entry point
evaluate.py   Batch extractor evaluation against label_data/
setup_env.sh  Creates .venv and installs all dependencies
```
