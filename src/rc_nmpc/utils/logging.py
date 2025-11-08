from pathlib import Path
import csv
from typing import Dict, Any

class CsvLogger:
    def __init__(self, path: str, fieldnames):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()
    def write(self, row: Dict[str, Any]):
        self.writer.writerow(row); self.f.flush()
    def close(self):
        self.f.close()
