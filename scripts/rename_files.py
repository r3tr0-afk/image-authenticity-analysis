import os
from pathlib import Path

FOLDER = Path("../data/raw/fake")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

files = [p for p in FOLDER.iterdir() if p.suffix.lower() in EXTENSIONS]
files.sort()

for idx, file in enumerate(files, start=1):
    new_name = f"fake_{idx:06d}{file.suffix.lower()}"
    new_path = file.with_name(new_name)
    file.rename(new_path)

print("Renaming complete.")
