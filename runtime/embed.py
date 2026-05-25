"""
Embedding generation script for Rocky datasets.

Usage:
    python -m runtime.embed

Loads all JSONL files from output_datasets/, generates embeddings using
sentence-transformers/all-MiniLM-L6-v2, and stores them in ChromaDB.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Generate embeddings for all datasets and store in ChromaDB."""
    logger.info("Loading datasets...")
    from runtime.utils import load_all_datasets, get_text_for_entry, CHROMA_DB_DIR

    datasets = load_all_datasets()
    total_entries = sum(len(entries) for entries in datasets.values())
    logger.info(f"Loaded {total_entries} entries across {len(datasets)} datasets.")

    # Prepare documents, metadatas, and ids
    documents = []
    metadatas = []
    ids = []

    for dataset_type, entries in datasets.items():
        for idx, entry in enumerate(entries):
            text = get_text_for_entry(dataset_type, entry)
            if not text.strip():
                continue

            doc_id = f"{dataset_type}_{idx}"

            # Build metadata payload
            metadata = {
                "dataset_type": dataset_type,
                "chapter": str(entry.get("chapter", "")),
                "chunk_id": entry.get("source_metadata", {}).get("chunk_id", "")
                if isinstance(entry.get("source_metadata"), dict)
                else "",
                "source_file": f"rocky_{dataset_type}.jsonl",
            }

            # Add topic or knowledge_type if available
            if "topic" in entry:
                metadata["topic"] = str(entry["topic"])
            elif "knowledge_type" in entry:
                metadata["topic"] = str(entry["knowledge_type"])
            elif "behavior_type" in entry and entry["behavior_type"]:
                metadata["topic"] = str(entry["behavior_type"])
            else:
                metadata["topic"] = dataset_type

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

    logger.info(f"Preparing to embed {len(documents)} documents...")

    # Load embedding model
    logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Model loaded. Generating embeddings (this may take a while)...")

    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    logger.info(f"Generated {len(embeddings)} embeddings.")

    # Store in ChromaDB
    import chromadb

    chroma_db_path = str(CHROMA_DB_DIR)
    logger.info(f"Storing embeddings in ChromaDB at {chroma_db_path}...")

    client = chromadb.PersistentClient(path=chroma_db_path)

    # Delete existing collection if present to avoid duplicates
    try:
        client.delete_collection("rocky_all")
    except ValueError:
        pass  # Collection doesn't exist yet

    collection = client.create_collection(
        "rocky_all",
        metadata={"description": "All Rocky datasets embeddings"},
    )

    # Add in batches of 1000
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        logger.info(f"  Stored batch {i // batch_size + 1}: entries {i} to {end - 1}")

    logger.info("Done! Embeddings stored successfully.")
    logger.info(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()