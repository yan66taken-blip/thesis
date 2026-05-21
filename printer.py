import numpy as np
import pandas as pd

azure=pd.read_csv("cli_evaluation_log.csv")
print(azure["tokens_used"].dropna())