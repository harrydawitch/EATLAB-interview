# db/data.py
import os
import pandas as pd

def append_df_to_csv(df: pd.DataFrame, file_path: str):
    """
    Append a DataFrame to a CSV file, writing header
    if the file does not yet exist.
    """
    mode   = "a" if os.path.exists(file_path) else "w"
    header = not os.path.exists(file_path)
    df.to_csv(
        file_path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8"
    )
