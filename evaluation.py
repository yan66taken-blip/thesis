import pandas as pd
import ast

LOG_FILE = "cli_evaluation_log.csv"
AZURE_LABELS = "azure_label.csv"

ALL_FIELDS = [
    "service_category",
    "user_symptom_category",
    "root_cause_category"
]

def normalize_pred(x):
    if isinstance(x, str) and x.startswith("["):
        x = ast.literal_eval(x)  # convert string list → actual list
    if isinstance(x, list):
        return set(i.strip().upper() for i in x)
    return {str(x).strip().upper()}

def normalize_actual(x):
    if isinstance(x, str):
        return set(i.strip().upper() for i in x.split(","))
    return {str(x).strip().upper()}

def evaluate(field):
    log_result = pd.read_csv(LOG_FILE)
    azure_labels = pd.read_csv(AZURE_LABELS)
    label_field = "label_" + field
    if field=="user_symptom_category":
        pred = log_result[field].apply(normalize_pred)
        actual = azure_labels[label_field].apply(normalize_actual)
    else:
        pred = log_result[field].str.upper()
        actual = azure_labels[label_field].str.upper()

    accuracy = (pred == actual).mean()
    print(f"Accuracy for {field}: {accuracy:.2%}")
    matches = pred == actual
    mismatches = ~matches
    if mismatches.any():
        print(f"--- Mismatches for {field} ---")
        mismatch_df = pd.DataFrame({
            "predicted": log_result.loc[mismatches, field].str.upper(),
            "actual": azure_labels.loc[mismatches, label_field].str.upper()
        })

        print(mismatch_df)

        # Optional: save to CSV for inspection
        mismatch_df.to_csv(f"mismatch_{field}.csv", index=False)
    else:
        print("No mismatches 🎉")

if __name__ == "__main__":
    for field in ALL_FIELDS:
        evaluate(field)