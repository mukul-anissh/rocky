"""
Retrieval module for the Rocky conversational runtime.

Queries ChromaDB and returns top relevant entries with metadata and scores.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves context from ChromaDB based on user messages."""

    def __init__(self, collection_name: str = "rocky_all"):
        self.collection_name = collection_name
        self.collection = None
        self._load_collection()

    def _load_collection(self):
        """Connect to ChromaDB and load the collection."""
        import chromadb
        from runtime.utils import CHROMA_DB_DIR

        chroma_db_path = str(CHROMA_DB_DIR)
        client = chromadb.PersistentClient(path=chroma_db_path)
        try:
            self.collection = client.get_collection(self.collection_name)
        except ValueError:
            logger.warning(
                f"Collection '{self.collection_name}' not found in ChromaDB. "
                "Run `python -m runtime.embed` first."
            )
            self.collection = None

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        dataset_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the ChromaDB collection for relevant entries.

        Args:
            query_text: The text to search for.
            n_results: Number of results to return.
            dataset_type: Optional filter for a specific dataset type.

        Returns:
            List of dicts with keys: text, metadata, similarity_score
        """
        if not self.collection:
            logger.warning("No collection loaded. Run `python -m runtime.embed` first.")
            return []

        # Build query kwargs
        query_kwargs = {
            "query_texts": [query_text],
            "n_results": n_results,
        }

        if dataset_type:
            query_kwargs["where"] = {"dataset_type": dataset_type}

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # Format results
        formatted = []
        if not results["ids"]:
            return []

        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0),
            })

        return formatted

    def query_all_datasets(
        self, query_text: str, results_per_dataset: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query across all dataset types individually for balanced results.

        Args:
            query_text: The text to search for.
            results_per_dataset: Number of results per dataset type.

        Returns:
            Dict mapping dataset_type -> list of results.
        """
        dataset_types = [
            "dialogues", "memories", "knowledge",
            "behavior", "relationships", "chunks"
        ]

        results = {}
        for dtype in dataset_types:
            hits = self.query(query_text, n_results=results_per_dataset, dataset_type=dtype)
            if hits:
                results[dtype] = hits

        return results