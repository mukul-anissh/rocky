"""Utility helpers for the Rocky conversational runtime."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Generator


OUTPUT_DATASETS_DIR = Path(__file__).resolve().parent.parent / "output_datasets"
CHROMA_DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    if not filepath.exists():
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Load all JSONL dataset files from output_datasets/."""
    dataset_files = {
        "chunks": OUTPUT_DATASETS_DIR / "rocky_chunks.jsonl",
        "dialogues": OUTPUT_DATASETS_DIR / "rocky_dialogues.jsonl",
        "memories": OUTPUT_DATASETS_DIR / "rocky_memories.jsonl",
        "knowledge": OUTPUT_DATASETS_DIR / "rocky_knowledge.jsonl",
        "behavior": OUTPUT_DATASETS_DIR / "rocky_behavior.jsonl",
        "relationships": OUTPUT_DATASETS_DIR / "rocky_relationships.jsonl",
    }
    datasets = {}
    for key, path in dataset_files.items():
        datasets[key] = load_jsonl(path)
    return datasets


def get_text_for_entry(dataset_type: str, entry: Dict[str, Any]) -> str:
    """Extract the primary text content from a dataset entry for embedding."""
    if dataset_type == "chunks":
        return entry.get("text", "")
    elif dataset_type == "dialogues":
        dialogue = entry.get("dialogue", "")
        context = entry.get("scene_context", "")
        response = entry.get("grace_response", "")
        parts = [p for p in [context, dialogue, response] if p]
        return " | ".join(parts)
    elif dataset_type == "memories":
        summary = entry.get("event_summary", "")
        knowledge = " ".join(entry.get("knowledge_learned", []))
        return f"{summary} {knowledge}".strip()
    elif dataset_type == "knowledge":
        return entry.get("statement", "")
    elif dataset_type == "behavior":
        return entry.get("observation", "")
    elif dataset_type == "relationships":
        return entry.get("interaction_summary", "")
    return json.dumps(entry)