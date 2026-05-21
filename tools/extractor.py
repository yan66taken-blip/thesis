from langchain_core.tools import tool
from openai import OpenAI
from pydantic import BaseModel
from typing import Union
import pandas as pd
import os
import time

STORE_PATH = "extractor_log.csv"

# -----------------------
# 1. Schema
# -----------------------
class Metadata(BaseModel):
   vendor: str
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
# 4. Tool
# -----------------------
@tool
def extractor(report: str) -> dict:
    """
    Extract structured metadata from an incident report and save it to the store.
    Returns the extracted record as a dict.
    """
    start = time.time()
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Incident report:\n{report}"}
        ],
        response_format=Metadata,
        temperature=0,
    )
    elapsed = time.time() - start

    record = response.choices[0].message.parsed.model_dump()
    tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

    # Normalize user_symptom_category to string for CSV storage
    if isinstance(record.get("user_symptom_category"), list):
        record["user_symptom_category"] = ",".join(record["user_symptom_category"])

    record["time_sec"] = round(elapsed, 3)
    record["tokens_used"] = tokens_used

    # Append to CSV store (create with header if it doesn't exist yet)
    df_new = pd.DataFrame([record])
    if os.path.exists(STORE_PATH):
        df_new.to_csv(STORE_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(STORE_PATH, mode="w", header=True, index=False)

    return record
