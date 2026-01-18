import csv
import json
import sys
from pathlib import Path


def csv_to_json(csv_path, json_path):
    csv_path = Path(csv_path)
    json_path = Path(json_path)

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 去掉空白
            clean = {k: (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()}
            rows.append(clean)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Converted {csv_path} -> {json_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json.py input.csv output.json")
        sys.exit(1)

    csv_to_json(sys.argv[1], sys.argv[2])
