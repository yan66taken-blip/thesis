# Final Extraction Evaluation

## Reproducing these results

```bash
bash setup_env.sh              # creates .venv, installs dependencies (run once)
source .venv/bin/activate
export OPENAI_API_KEY=sk-...   # needs access to gpt-4o-mini

python eval_final/reproduce.py            # all three vendors
python eval_final/reproduce.py Azure      # single vendor
```

This sends the **full, untruncated** report text — it deliberately does not go
through `tools.extractor.extract_record` (which defaults to
`MAX_REPORT_CHARS = 4000`), to exactly match the methodology below. It reads
whatever is currently in `prompt/*.txt`, so results will drift if those files
change after this was written (Jul 2026). Expect run-to-run variance of a few
percentage points even with `temperature=0` — this is normal LLM API
non-determinism, not a bug.


Model: **gpt-4o-mini**, with the following production prompt changes applied:
- `prompt/root_cause_instruction.txt` — CONFIG/DEPLOY disambiguated by *what* was wrong (setting vs. code/hardware), not by surface words like "configuration"/"deployed". Root fix: model over-predicted CONFIG whenever the literal word "configuration" appeared, and misclassified standalone hardware faults as OTHERS instead of DEPLOY.
- `prompt/user_symp_instruction.txt` — UNAVIL vs. ERROR disambiguated by whether the connection itself succeeded, not by whether the report's wording contains "error"/"failure". Root fix: model defaulted to ERROR whenever the text said "connection failure"/"timeout", even when the report was describing a total-inaccessibility (UNAVIL) scenario.
- `prompt/prompt_template.txt` — added 2 few-shot examples (CONFIG-via-pushed-config, DEPLOY-via-hardware-fault) targeting the same two confusions, on top of the original 2. Disproportionately helps smaller models (gpt-4o-mini gained further from these on top of the instruction-text fixes alone).

No input truncation applied (full report text sent).

## Results (n=460 total: AWS 150, Azure 95, GCP 215)

| Field | AWS | Azure | GCP |
|---|---|---|---|
| service_category | 86.0% | 72.1% | 59.5% |
| user_symptom_category | 77.3% | 49.5% | 48.0% |
| root_cause_category | N/A (no GT) | 65.2% | 64.7% |
| start_time | 94.6% | ~98.9% | 78.8% |
| end_time | 86.6% | ~95.7% | 81.6% |
| latency mean | 1.79s | 2.54s | 1.85s |

## Why gpt-4o-mini over gpt-4o

gpt-4o scored higher on root_cause_category (73.9-75.0% vs. 65.2-64.7%) but at ~3x the latency (6.9s vs. 2.2-2.5s mean) and higher per-call cost. gpt-4o-mini stays under the thesis's NFR4 latency threshold (3s) even without truncation, while gpt-4o does not. Given the system is designed for interactive/agentic use (not one-shot final reports — Section 6 already proposes a human-correction mechanism for engineers to fix wrong fields), the latency/cost savings were judged to outweigh the accuracy gap, especially after the prompt fixes closed part of that gap (root_cause_category on gpt-4o-mini improved from a 60.9% starting point to 65.2% via the three prompt changes above).

## Other models tested during iteration (data not retained, summary only)

- **gpt-4.1**: dominated by gpt-4o on every field tested (Azure: root_cause 71.7% vs 75.0%, service_category 70.9% vs 73.3%, service_name 62.1% vs 65.5%) at slightly higher latency — no scenario where it was the better choice.
- **gpt-4.1-mini**: showed a real regression on root_cause_category (-5.4pp) when the same root_cause prompt fix was applied, unlike gpt-4o and gpt-4o-mini which both took the fix cleanly. Not used further.
- **gpt-3.5-turbo**: incompatible — does not support OpenAI's Structured Outputs (`response_format` json_schema) needed for the extraction schema.
- **Input truncation (4000 chars)**: tested and validated as latency-neutral-to-positive on accuracy (root_cause_category held steady, zero net accuracy loss even on the longest reports). Not used in the final system — decided against it to keep the pipeline simple and match the no-truncation methodology above.
- **GCP timestamp preprocessing** (stripping the repeated `created`/`modified`/`when` JSON metadata down to just the human-readable `text` fields): tested and **rejected** — caused a regression (start_time 66.8% → 38.6%) because those timestamps often serve as the only date anchor for relative time references inside the update text, not pure noise.
