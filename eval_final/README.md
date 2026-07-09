# Final Extraction Evaluation

This directory is the evidence trail for one design decision in the ODAgent
thesis: **which model and prompt configuration the Extraction Agent should
use.** It holds the final numbers, the charts, and a condensed record of what
else was tried and why it was rejected — enough to write the "why we chose
this" section of the thesis without needing to re-run anything.

## TL;DR

**Model: gpt-4o-mini.** Production prompt in `prompt/` has three changes on
top of the version described in the thesis text: a clarified CONFIG-vs-DEPLOY
rule, a clarified UNAVIL-vs-ERROR rule, and two extra few-shot examples
targeting both. No input truncation.

## Results (n=460: AWS 150, Azure 95, GCP 215)

| Field | AWS | Azure | GCP |
| --- | --- | --- | --- |
| service_category | 86.7% | 73.3% | 58.6% |
| user_symptom_category | 77.3% | 44.2% | 47.4% |
| root_cause_category | N/A (no GT) | 64.1% | 67.6% |
| start_time | 96.0% | 100.0% | 80.4% |
| end_time | 87.9% | 95.7% | 83.7% |
| latency mean | 1.79s | 2.05s | 1.82s |
| tokens (avg total / cached) | 3511 / 2912 | 4385 / 2913 | 4531 / 2944 |

Charts: `latency.png` (per-vendor latency distribution), `token_usage.png`
(per-vendor average token usage, total vs. cached).

Raw per-row data: `AWS.csv`, `Azure.csv`, `GCP.csv` — predictions, ground
truth, match flags, latency, and token columns, one row per incident report.

## The three prompt changes, and why

- **`prompt/root_cause_instruction.txt`** — CONFIG vs. DEPLOY is now
  disambiguated by *what* was actually wrong (a setting vs. code/hardware),
  not by surface words. The old prompt caused the model to default to CONFIG
  whenever the literal word "configuration" appeared anywhere in the report,
  and to misclassify standalone hardware faults as OTHERS instead of DEPLOY.
- **`prompt/user_symp_instruction.txt`** — UNAVIL vs. ERROR is now
  disambiguated by whether the connection itself succeeded, not by whether
  the report's wording contains "error"/"failure". The old prompt caused the
  model to default to ERROR whenever the text said "connection
  failure"/"timeout", even when the report was describing total
  inaccessibility (UNAVIL).
- **`prompt/prompt_template.txt`** — two extra few-shot examples were added
  (a CONFIG-via-pushed-config case, a DEPLOY-via-hardware-fault case) on top
  of the original two, targeting the same two confusions. This
  disproportionately helped the smaller model — gpt-4o-mini gained further
  accuracy from the examples on top of what the instruction-text fixes alone
  provided.

Together, these took gpt-4o-mini's root_cause_category from a 60.9% starting
point up to 64-68%, and user_symptom_category up by double digits on Azure
and AWS, with no measured regression elsewhere.

## Why gpt-4o-mini instead of gpt-4o

gpt-4o scores higher on root_cause_category (73.9-75.0% vs. 64.1-67.6% here)
but at roughly 3x the latency (~6.9s vs. ~1.8-2.1s mean) and higher per-call
cost. gpt-4o-mini stays comfortably under the thesis's NFR4 latency threshold
(3s) even without truncation; gpt-4o does not. The system is designed for
interactive/agentic use rather than one-shot final reports — Section 6
already proposes a human-correction mechanism for engineers to fix wrong
fields — so the latency and cost savings were judged to outweigh the
remaining accuracy gap, especially once the prompt fixes above closed part
of it.

## Alternatives tried and rejected (data not retained)

| Tried | Result |
| --- | --- |
| gpt-4.1 | Dominated by gpt-4o on every field tested (Azure: root_cause 71.7% vs 75.0%, service_category 70.9% vs 73.3%, service_name 62.1% vs 65.5%) at slightly *higher* latency. No scenario where it was the better choice. |
| gpt-4.1-mini | Took the root_cause prompt fix badly — a real regression (-5.4pp), unlike gpt-4o and gpt-4o-mini which both improved from the same change. Not used further. |
| gpt-3.5-turbo | Incompatible — doesn't support OpenAI's Structured Outputs (`response_format` json_schema), which the extraction schema requires. |
| Input truncation (4000 chars) | Validated as latency-neutral-to-positive with zero net accuracy loss, even on the longest reports. Not used in the final system anyway — decided against it to keep the pipeline simple. |
| GCP timestamp preprocessing (stripping the repeated `created`/`modified`/`when` JSON metadata down to just the `text` fields) | **Rejected** — caused a regression (start_time 66.8% → 38.6%). Those timestamps often serve as the only date anchor for relative time phrases inside the update text; they aren't pure noise. |

## Reproducing these results

```bash
bash setup_env.sh              # creates .venv, installs dependencies (run once)
source .venv/bin/activate
export OPENAI_API_KEY=sk-...   # needs access to gpt-4o-mini
export BENCH_MODEL=gpt-4o-mini

python evaluate.py             # runs all three vendors, writes eval/<vendor>_label/
```

`evaluate.py` sends the full, untruncated report text by default —
`tools/extractor.py` has no truncation logic, per the decision above. It
reads whatever is currently in `prompt/*.txt`, so results will drift if
those files change after this was written (Jul 2026). Expect run-to-run
variance of a few percentage points even at `temperature=0` — that's normal
LLM API non-determinism at this sample size, not a bug. Two full reruns
during this evaluation stayed within ±1-3pp of each other on every field.
