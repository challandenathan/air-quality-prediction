from pathlib import Path
import pandas as pd

def load_data(file_path: str):
    """
    Load all beijing air quality csv files into a single data frame
    """
    file_path = Path(file_path)
    csv_files = sorted(file_path.glob("*.csv"))
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data