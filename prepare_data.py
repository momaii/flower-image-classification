import zipfile
from pathlib import Path

# ==============================
# Configuration
# ==============================

PROJECT_ROOT = Path(__file__).parent
DATA_ZIP = PROJECT_ROOT / "dataset.zip"
DATA_DIR = PROJECT_ROOT / "data"


# ==============================
# Functions
# ==============================

def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def main():
    if not DATA_ZIP.exists():
        print("dataset.zip not found in project root.")
        print("Please place the zip file next to prepare_data.py.")
        return

    DATA_DIR.mkdir(exist_ok=True)

    extract_zip(DATA_ZIP, DATA_DIR)

    print(f"Dataset ready in '{DATA_DIR.name}/' directory.")


if __name__ == "__main__":
    main()
