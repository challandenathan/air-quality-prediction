from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_data(file_path: str):
    """
    Load all beijing air quality csv files into a single data frame
    """
    file_path = PROJECT_ROOT / file_path
    csv_files = sorted(file_path.glob("*.csv")) # *.csv means "finds every csv file"
    dfs = []
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {file_path}")
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data