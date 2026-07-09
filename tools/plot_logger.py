import os
import pandas as pd
from datetime import datetime, timezone

PLOT_LOG_PATH = os.path.join("logs", "plot_log.csv")

COLUMNS = ["timestamp", "tool", "vendors", "years", "output_path", "record_count", "time_sec"]


def log_plot(tool: str, vendors, years, output_path: str, record_count: int, time_sec: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool,
        "vendors": ",".join(sorted(vendors)) if vendors else "ALL",
        "years": ",".join(str(y) for y in sorted(years)) if years else "ALL",
        "output_path": output_path,
        "record_count": record_count,
        "time_sec": round(time_sec, 3),
    }
    df_new = pd.DataFrame([entry])
    if os.path.exists(PLOT_LOG_PATH):
        df = pd.concat([pd.read_csv(PLOT_LOG_PATH), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(PLOT_LOG_PATH, index=False)
