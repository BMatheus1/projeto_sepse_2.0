from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .anonymization import anonymize_text
from .config import DATASET_SUMMARY_PATH, FINE_TUNING_DATASET_PATH, RAW_DATA_DIR, SYSTEM_PROMPT
from .preprocessing import clean_text, validate_record


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"source": path.stem, "input": "", "output": "", "_error": f"Linha {line_number}: {exc}"}


def build_chat_record(record: Dict[str, Any]) -> Dict[str, Any]:
    user_content = anonymize_text(clean_text(record["input"]))
    assistant_content = anonymize_text(clean_text(record["output"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": clean_text(record["source"]),
            "synthetic": True,
        },
    }


def prepare_dataset(raw_dir: Path = RAW_DATA_DIR, output_path: Path = FINE_TUNING_DATASET_PATH) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DATASET_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    valid = 0
    discarded = 0
    sources: List[str] = []
    prepared: List[Dict[str, Any]] = []

    for path in sorted(raw_dir.glob("*.jsonl")):
        for record in iter_jsonl(path):
            total += 1
            is_valid, _errors = validate_record(record)
            if not is_valid:
                discarded += 1
                continue
            prepared.append(build_chat_record(record))
            valid += 1
            source = str(record.get("source", path.stem))
            if source not in sources:
                sources.append(source)

    with output_path.open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "records_read": total,
        "valid_records": valid,
        "discarded_records": discarded,
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(output_path),
    }
    DATASET_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = prepare_dataset()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
