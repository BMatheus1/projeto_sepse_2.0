from __future__ import annotations

import json
import random
import hashlib
from collections import Counter, defaultdict
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
    # Preserve section breaks that the model should learn to generate.
    normalize = lambda text: "\n".join(clean_text(line) if line.strip() else "" for line in text.splitlines()).strip()
    user_content = anonymize_text(normalize(record["input"]))
    assistant_content = anonymize_text(normalize(record["output"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": clean_text(record["source"]),
            "synthetic": True,
            "category": record.get("category", "legacy"),
            "group_id": record.get("group_id", "legacy:" + record["source"]),
        },
    }


def normalized_text(text):
    return " ".join(text.casefold().split())


def split_records(records, seed=42):
    """Approximate 80/10/10 by whole families, never by individual paraphrases.

    Legacy examples were seen by experiment 01 and remain train-only. Families
    sharing an identical user or assistant text are merged before assignment.
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[row["metadata"]["group_id"]].append(row)
    parents = {key: key for key in grouped}
    def find(key):
        while parents[key] != key:
            key = parents[key]
        return key
    seen = {}
    for key, rows in grouped.items():
        for row in rows:
            for message in row["messages"]:
                if message["role"] == "system":
                    continue
                signature = (message["role"], normalized_text(message["content"]))
                if signature in seen:
                    parents[find(key)] = find(seen[signature])
                seen[signature] = key
    components = defaultdict(list)
    for key, rows in grouped.items():
        components[find(key)].extend(rows)
    legacy, candidates = [], []
    for key, rows in sorted(components.items()):
        if any(r["metadata"]["category"] == "legacy" for r in rows):
            legacy.extend(rows)
        else:
            candidates.append((key, rows))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    splits = {"train": list(legacy), "validation": [], "test": []}
    target = round(len(records) * 0.1)
    # Bounded subset sum chooses a closest achievable target without splitting families.
    for name in ("validation", "test"):
        # Ensure held-out families cover the main alignment goals, not only clinical facts.
        priority = ["fontes", "validacao_humana", "prompt_injection", "recusas", "triagem", "exames", "alertas"]
        buckets = defaultdict(list)
        for item in candidates:
            buckets[item[1][0]["metadata"]["category"]].append(item)
        order = priority + sorted(set(buckets) - set(priority))
        candidates = [buckets[category][i] for i in range(max((len(b) for b in buckets.values()), default=0))
                      for category in order if i < len(buckets[category])]
        choices = {0: []}
        for i, (_, rows) in enumerate(candidates):
            for count, indices in list(choices.items()):
                new_count = count + len(rows)
                if new_count <= target + max(len(rows), 1) and new_count not in choices:
                    choices[new_count] = indices + [i]
        size = min(choices, key=lambda count: (abs(count - target), count > target))
        chosen = set(choices[size])
        for i in chosen:
            splits[name].extend(candidates[i][1])
        candidates = [item for i, item in enumerate(candidates) if i not in chosen]
    for _, rows in candidates:
        splits["train"].extend(rows)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Byte-stable across Windows and Colab, including the dataset SHA-256.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(raw_dir: Path = RAW_DATA_DIR, output_path: Path = FINE_TUNING_DATASET_PATH,
                    summary_path: Path = DATASET_SUMMARY_PATH, seed: int = 42) -> Dict[str, Any]:
    total = discarded = duplicates = 0
    prepared = []
    seen = set()
    for path in sorted(raw_dir.glob("*.jsonl")):
        for record in iter_jsonl(path):
            total += 1
            is_valid, _ = validate_record(record)
            if not is_valid:
                discarded += 1
                continue
            row = build_chat_record(record)
            signature = tuple((m["role"], normalized_text(m["content"])) for m in row["messages"])
            if signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            prepared.append(row)
    splits = split_records(prepared, seed=seed)
    write_jsonl(output_path, prepared)
    split_info = {}
    for name, rows in splits.items():
        path = output_path.parent / f"fine_tuning_{name}.jsonl"
        write_jsonl(path, rows)
        split_info[name] = {"count": len(rows), "groups": len({r["metadata"]["group_id"] for r in rows}),
                            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "categories": dict(Counter(r["metadata"]["category"] for r in rows))}
    summary = {"records_read": total, "valid_records": len(prepared), "discarded_records": discarded,
               "duplicates_removed": duplicates, "sources": sorted({r["metadata"]["source"] for r in prepared}),
               "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_path": str(output_path),
               "seed": seed, "split_target": {"train": 0.8, "validation": 0.1, "test": 0.1},
               "split_policy": "whole_families; alignment categories prioritized; identical user/assistant texts merged; legacy train-only",
               "splits": split_info}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(prepare_dataset(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
