from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Union
import pandas as pd
import os

STORE_PATH = "cli_evaluation_log.csv"

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

parser = PydanticOutputParser(pydantic_object=Metadata)


# -----------------------
# 2. LLM
# -----------------------
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -----------------------
# 3. Prompt template
# -----------------------
def generate_prompt() -> str:
    service_category_lst = ['COMPUTE', 'STORAGE', 'NETWORK', 'SECURITY', 'AI', 'MANAGEMENT', 'ANALYTICS', 'DATABASE', 'OTHERS', 'UNKNOWN']
    user_symp_lst = ['ERROR', 'UNAVIL', 'DELAY', 'DEPERF', 'OTHERS', 'UNKNOWN']
    user_symp_instruction = open('prompt/user_symp_instruction.txt').read()
    root_cause_lst = ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN']
    root_cause_instruction = open('prompt/root_cause_instruction.txt').read()
    prompt_template = open('prompt/prompt_template.txt').read()
    prompt = prompt_template.format(
        service_category_lst=service_category_lst,
        user_symp_lst=user_symp_lst,
        root_cause_lst=root_cause_lst,
        user_symp_instruction=user_symp_instruction,
        root_cause_instruction=root_cause_instruction
    )
    return prompt

prompt = generate_prompt()
prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("human", "Incident report:\n{report}")
])


# -----------------------
# 4. Build chain
# -----------------------
chain = prompt_template | chat | parser


# -----------------------
# 5. Tool
# -----------------------
@tool
def extractor(report: str) -> dict:
    """
    Extract structured metadata from an incident report and save it to the store.
    Returns the extracted record as a dict.
    """
    result = chain.invoke({"report": report})
    record = result.model_dump()

    # Normalize user_symptom_category to string for CSV storage
    if isinstance(record.get("user_symptom_category"), list):
        record["user_symptom_category"] = ",".join(record["user_symptom_category"])

    # Append to CSV store (create with header if it doesn't exist yet)
    df_new = pd.DataFrame([record])
    if os.path.exists(STORE_PATH):
        df_new.to_csv(STORE_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(STORE_PATH, mode="w", header=True, index=False)

    return record