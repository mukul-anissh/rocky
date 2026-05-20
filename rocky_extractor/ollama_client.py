import json
import time
import logging
from typing import Type, Optional
from pydantic import BaseModel, ValidationError
import ollama
from rocky_extractor.cache import LLMCache

logger = logging.getLogger(__name__)

class OllamaExtractionClient:
    """Manages connection, caching, validation, and retries for local Ollama extraction."""
    def __init__(
        self, 
        model: str = "qwen3:8b", 
        temperature: float = 0.2, 
        top_p: float = 0.8, 
        repeat_penalty: float = 1.1,
        use_cache: bool = True,
        cache_db_path: str = ".rocky_cache.db"
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.use_cache = use_cache
        self.cache = LLMCache(cache_db_path) if use_cache else None

        # Check and ensure the model exists locally
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Verifies if the target model is present, and pulls it if missing."""
        try:
            logger.info(f"Verifying availability of Ollama model '{self.model}'...")
            models_list = ollama.list()
            local_models = [m.get("model") or m.get("name") for m in models_list.get("models", [])]
            
            # Ollama lists names with tags sometimes (e.g. qwen3:8b or qwen3:latest)
            model_exists = False
            for m in local_models:
                if m == self.model or m.startswith(f"{self.model}:") or self.model.startswith(f"{m}:"):
                    model_exists = True
                    break

            if not model_exists:
                logger.info(f"Model '{self.model}' not found locally. Pulling from Ollama registry...")
                ollama.pull(self.model)
                logger.info(f"Model '{self.model}' pulled successfully.")
            else:
                logger.info(f"Model '{self.model}' is available.")
        except Exception as e:
            logger.warning(
                f"Could not contact Ollama to verify/pull model: {e}. "
                "Ensure Ollama is running (`ollama serve`). Proceeding assuming model exists..."
            )

    def extract_structured_data(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema_model: Type[BaseModel],
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> Optional[BaseModel]:
        """
        Executes the extraction prompt with Ollama, validates it against schema_model,
        handles caching, retries, and malformed JSON.
        """
        # Step 1: Check cache if enabled
        if self.use_cache and self.cache:
            cached_val = self.cache.get(self.model, system_prompt, user_prompt)
            if cached_val:
                try:
                    parsed = json.loads(cached_val)
                    validated = schema_model.model_validate(parsed)
                    logger.debug("Successfully validated from Cache!")
                    return validated
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Cached data invalid for current schema. Re-querying LLM: {e}")

        # Step 2: Query LLM with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Querying Ollama (Attempt {attempt}/{max_retries})...")
                
                start_time = time.time()
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "repeat_penalty": self.repeat_penalty
                    },
                    format="json"  # Forces JSON response from Ollama
                )
                duration = time.time() - start_time
                logger.debug(f"Ollama response received in {duration:.2f}s")

                raw_content = response["message"]["content"]
                if not raw_content or not raw_content.strip():
                    raise ValueError("Ollama returned an empty response.")

                # Try parsing JSON
                try:
                    parsed_json = json.loads(raw_content)
                except json.JSONDecodeError as je:
                    logger.warning(f"Failed to parse Ollama JSON response: {je}\nRaw content: {raw_content}")
                    raise je

                # Try Pydantic Validation
                try:
                    validated_data = schema_model.model_validate(parsed_json)
                    
                    # Successfully validated! Store in cache
                    if self.use_cache and self.cache:
                        self.cache.set(self.model, system_prompt, user_prompt, raw_content)
                        
                    return validated_data
                except ValidationError as ve:
                    logger.warning(f"JSON schema validation failed on attempt {attempt}: {ve}")
                    raise ve

            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Failed to extract structured data after {max_retries} attempts: {e}")
                    raise e
                
                sleep_time = backoff_factor ** attempt
                logger.info(f"Retrying in {sleep_time:.2f} seconds due to error: {e}")
                time.sleep(sleep_time)

        return None
