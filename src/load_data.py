from pathlib import Path
import pandas as pd


def load_transactions(file_path: str) -> pd.DataFrame:
    """
    Load transaction data from a CSV file.

    Args:
        file_path: Path to the transaction dataset.

    Returns:
        A pandas DataFrame containing transaction records.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(path)
    return df