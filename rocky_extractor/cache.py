import sqlite3
import hashlib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMCache:
    """A lightweight, disk-based SQLite cache for LLM request/response pairs."""
    def __init__(self, db_path: str = ".rocky_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        """Returns a thread-safe connection to the SQLite database."""
        # Use isolation_level=None for autocommit or manage transactions explicitly
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the cache table if it does not already exist."""
        logger.debug(f"Initializing SQLite cache at {self.db_path}")
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _generate_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Generates a unique MD5 hash for the model and prompts combination."""
        combined = f"{model}:{system_prompt}:{user_prompt}".encode("utf-8")
        return hashlib.md5(combined).hexdigest()

    def get(self, model: str, system_prompt: str, user_prompt: str) -> str | None:
        """
        Retrieves a cached response if it exists.
        Returns the cached string, or None if not found.
        """
        key = self._generate_key(model, system_prompt, user_prompt)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", 
                (key,)
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Cache hit for key {key[:8]}...")
                return row["response"]
                
        logger.debug(f"Cache miss for model {model}")
        return None

    def set(self, model: str, system_prompt: str, user_prompt: str, response: str):
        """Caches an LLM response with its corresponding request details."""
        key = self._generate_key(model, system_prompt, user_prompt)
        prompt_hash = hashlib.md5(system_prompt.encode("utf-8")).hexdigest()
        chunk_hash = hashlib.md5(user_prompt.encode("utf-8")).hexdigest()
        
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache (key, model, prompt_hash, chunk_hash, response, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, model, prompt_hash, chunk_hash, response, datetime.now().isoformat())
                )
                conn.commit()
                logger.debug(f"Cached response under key {key[:8]}...")
            except sqlite3.Error as e:
                logger.error(f"Failed to write to SQLite cache: {e}")

    def clear(self):
        """Clears all records in the cache database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()
        logger.info("LLM cache cleared successfully.")
