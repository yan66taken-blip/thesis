import time
import re
import json
import pandas as pd
from tools.extractor import extractor

LOG_FILE = "cli_evaluation_log.csv"

ALL_FIELDS = [
    "vendor",
    "service_name",
    "service_category",
    "start_time",
    "end_time",
    "user_symptom",
    "user_symptom_category",
    "root_cause",
    "root_cause_category"
]

from openai import OpenAI

client = OpenAI()

def select_fields(report, fields):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a field selector. Output ONLY a JSON array of field names. No objects. No values. No explanation."
            },
            {
                "role": "user",
                "content": f"""
            Allowed fields:
            {fields}

            Report:
            {report}

            Return only relevant fields from the allowed list.
            """
            }
        ]
    )
    print("Field selection response:", response.choices[0].message.content)
    cleaned = re.sub(r"```json|```", "", response.choices[0].message.content).strip()

    return json.loads(cleaned)

def information_response(Requirement, data):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate a clear, human-readable response based on the requirement and data."
            },
            {
                "role": "user",
                "content": f"""
Requirement:
{Requirement}

Data:
{data}
"""
            }
        ]
    )

    content = response.choices[0].message.content
    print("Information extraction response:", content)

    return content
   

def run_cli():
    print("=== AI Extraction CLI (Schema-Controlled) ===")

    while True:
        print("\n--- New Task ---")

        report = input("Hey, what report do you have?\n> ")

        # 1. AI selects fields (from predefined list)
        
        require = input(f"Which kind of info do you want to extract?  \n> ")

        selected_fields = select_fields(report, ALL_FIELDS)

        # 2. Extraction (ALL fields)
        start_time = time.time()

        result = extractor.invoke({"report": report})

        elapsed_time = time.time() - start_time

        # 3. Filter only selected fields
        filtered_result = {
            k: result.get(k, None) for k in selected_fields
        }

        information = information_response(require, filtered_result)

        print(information)
        # 5. Log
        log_entry = {
            "report": report,
            "vendor": result.get("vendor", None),
            "service_name": result.get("service_name", None),
            "service_category": result.get("service_category", None),   
            "start_time": result.get("start_time", None),
            "end_time": result.get("end_time", None),
            "user_symptom": result.get("user_symptom", None),
            "user_symptom_category": result.get("user_symptom_category", None),
            "root_cause": result.get("root_cause", None),
            "root_cause_category": result.get("root_cause_category", None),
            "time_sec": elapsed_time,
        }

        df_new = pd.DataFrame([log_entry])

        try:
            df_existing = pd.read_csv(LOG_FILE)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_all = df_new

        df_all.to_csv(LOG_FILE, index=False)

        print("\nSaved.")

        # 6. Stats
        print("\n--- Stats ---")
        print(f"Runs: {len(df_all)}")
        print(f"Avg time: {df_all['time_sec'].mean():.2f}s")

        if input("\nContinue? (y/n)\n> ").lower() != "y":
            break


if __name__ == "__main__":
    run_cli()