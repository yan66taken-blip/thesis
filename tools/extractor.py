from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Union



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
   user_symptom_category:  Union[str, list[str]] 
   root_cause: str
   root_cause_category: str

parser = PydanticOutputParser(pydantic_object=Metadata)


# -----------------------
# 2. LLM
# -----------------------
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -----------------------
# 3. Prompt template (IMPORTANT FIX)
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
    print("Generated prompt:%s", prompt)
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
# 5. Tool (REAL version)
# -----------------------
@tool
def extractor(report: str) -> dict:
    """
    Extract data from incident report.
    """

    result = chain.invoke({"report": report})

    return result.model_dump()