from langchain_core.tools import tool
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Union
import pandas as pd
import os
import time
import json
import threading

STORE_DIR = "logs"


# -----------------------
# 1. Schema
# -----------------------
class Metadata(BaseModel):
    vendor: Literal["Azure", "GCP", "AWS"]
    service_name: str
    service_category: str
    start_time: str
    end_time: str
    user_symptom: str
    user_symptom_category: Union[str, list[str]]
    root_cause: str
    root_cause_category: str


# -----------------------
# 2. Client
# -----------------------
MODEL = os.environ.get("BENCH_MODEL", "gpt-4o-mini")
client = OpenAI()


# -----------------------
# 3. System prompt
# -----------------------
def generate_prompt() -> str:
    service_category_lst = ['COMPUTE', 'STORAGE', 'NETWORK', 'SECURITY', 'AI', 'MANAGEMENT', 'ANALYTICS', 'DATABASE', 'OTHERS', 'UNKNOWN']
    user_symp_lst = ['ERROR', 'UNAVIL', 'DELAY', 'DEPERF', 'OTHERS', 'UNKNOWN']
    user_symp_instruction = open('prompt/user_symp_instruction.txt').read()
    root_cause_lst = ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN']
    root_cause_instruction = open('prompt/root_cause_instruction.txt').read()
    prompt_template = open('prompt/prompt_template.txt').read()
    return prompt_template.format(
        service_category_lst=service_category_lst,
        user_symp_lst=user_symp_lst,
        root_cause_lst=root_cause_lst,
        user_symp_instruction=user_symp_instruction,
        root_cause_instruction=root_cause_instruction
    )

system_prompt = generate_prompt()


# -----------------------
# 4. Store helpers
# -----------------------
def _append_and_sort(record: dict, store_dir: str) -> None:
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"{record['vendor']}.csv")
    df_new = pd.DataFrame([record])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    else:
        df = df_new

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df = df.sort_values("start_time").reset_index(drop=True)
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(path, index=False)


def clear_store() -> None:
    if os.path.isdir(STORE_DIR):
        for f in os.listdir(STORE_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(STORE_DIR, f))


# -----------------------
# 5. Core extraction (store_dir is explicit — used by evaluate.py too)
# -----------------------
def extract_record(report: str, store_dir: str, max_retries: int = 5) -> dict:
    start = time.time()
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Incident report:\n{report}"}
                ],
                response_format=Metadata,
            )
            if not MODEL.startswith("gpt-5"):
                kwargs["temperature"] = 0
            response = client.beta.chat.completions.parse(**kwargs)
            break
        except Exception as e:
            retryable = "429" in str(e) or "Connection error" in str(e) or "ConnectError" in str(e)
            if retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise
    elapsed = time.time() - start

    record = response.choices[0].message.parsed.model_dump()

    if isinstance(record.get("user_symptom_category"), list):
        record["user_symptom_category"] = ",".join(record["user_symptom_category"])

    cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0

    record["model"] = MODEL
    record["time_sec"] = round(elapsed, 3)
    record["tokens_used"] = response.usage.prompt_tokens + response.usage.completion_tokens
    record["cached_tokens"] = cached_tokens
    try:
        record["year"] = int(str(record["start_time"])[:4])
    except (ValueError, TypeError):
        record["year"] = None

    _append_and_sort(record, store_dir)

    return record


# -----------------------
# 6. Tools
# -----------------------
VALID_FIELDS = list(Metadata.model_fields.keys())


def _extract_partial(report: str, fields: list[str]) -> dict:
    """Quick LLM call for only the requested fields, no storage."""
    requested = [f for f in fields if f in VALID_FIELDS]
    if not requested:
        return {}
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract ONLY these fields from the incident report and return as JSON "
                    f"with exactly these keys: {', '.join(requested)}."
                ),
            },
            {"role": "user", "content": f"Incident report:\n{report}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


@tool
def extractor(report: str) -> dict:
    """
    Extract ALL structured metadata from an incident report and save it to the store.
    Returns the full extracted record as a dict.
    """
    return extract_record(report, STORE_DIR)


@tool
def smart_extractor(report: str, requested_fields: list[str]) -> dict:
    """
    Extract ONLY the fields the user asked about and return them immediately.
    A full extraction runs in the background to persist the complete record to the store.

    Args:
        report: Raw incident report text.
        requested_fields: Fields to extract for the user. Valid values:
            vendor, service_name, service_category, start_time, end_time,
            user_symptom, user_symptom_category, root_cause, root_cause_category.

    Returns:
        Dict containing only the requested fields.
    """
    thread = threading.Thread(target=extract_record, args=(report, STORE_DIR), daemon=True)
    thread.start()

    return _extract_partial(report, requested_fields)
