import time

from tools.extractor import extractor
from langchain_community.callbacks import get_openai_callback
import numpy as np
import pandas as pd

LOG_FILE = "cli_evaluation_log.csv"

azure=pd.read_csv("azure_label.csv")

len(azure)

accurate_service_category = 0
compared_catregory = "service_category"
original_category = "label_"+compared_catregory

for index, row in azure.iterrows():
    print(f"Processing row {index}...")
    report = row["description"]
    start_time = time.time()
    with get_openai_callback() as cb:
        result = extractor.invoke({"report": report})
    elapsed_time = time.time() - start_time
    tokens_used = cb.total_tokens
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
            "tokens_used": tokens_used,
        }
    
    df_new = pd.DataFrame([log_entry])

    try:
        df_existing = pd.read_csv(LOG_FILE)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all.to_csv(LOG_FILE, index=False)
   

accuracy = accurate_service_category / len(azure)
print(f"Accuracy for {compared_catregory}: {accuracy:.2%}")
